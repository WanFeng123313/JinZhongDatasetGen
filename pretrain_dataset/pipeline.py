from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from pathlib import Path

from pretrain_dataset.config import Config
from pretrain_dataset.llm_client import LLMClient, LLMError, TruncatedError
from pretrain_dataset.quality import (
    audit_jsonl,
    detach_extra_articles,
    detach_page_inserts,
    paper_skip_reason,
    record_reject_reason,
    salvage_record,
    tidy_page_markdown,
)
from pretrain_dataset.pdf_reparse import has_page_continuation, locate_pdf, render_pdf_pages
from pretrain_dataset.prompts import (
    build_chunk_messages,
    build_classify_messages,
    build_fast_messages,
    build_page_messages,
    build_single_pass_messages,
    build_stitch_messages,
)
from pretrain_dataset.text_norm import (
    cleanup_record_text,
    clean_preamble,
    clean_section_title,
    document_is_english,
    is_dropped_section,
    is_mostly_english,
    prepare_markdown,
    section_paragraphs,
    split_paragraph_groups,
    split_sections,
)

logger = logging.getLogger("pretrain_dataset")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.ok = 0
        self.skip = 0
        self.fail = 0
        self.records = 0
        self._lock = threading.Lock()

    def update(self, status: str, source: str, records: int = 0) -> None:
        with self._lock:
            self.done += 1
            setattr(self, status, getattr(self, status) + 1)
            self.records += records
            logger.info(
                "[%s] %s/%s 样本 %s %s",
                status,
                self.done,
                self.total,
                records,
                source,
            )


def discover_markdowns(config: Config) -> list[Path]:
    files = [
        path
        for path in config.input_dir.rglob("*.md")
        if "_logs" not in path.parts and path.is_file()
    ]
    files.sort()
    if config.match:
        files = [path for path in files if config.match in str(path)]
    if config.limit > 0:
        files = files[: config.limit]
    return files


def output_json_path(config: Config, markdown_path: Path) -> Path:
    relative = markdown_path.relative_to(config.input_dir).with_suffix(".json")
    return config.json_dir / relative


def estimate_max_tokens(char_len: int, cap: int) -> int:
    return min(cap, max(2048, int(char_len * 1.6) + 1024))


def _parse_json_object(text: str) -> dict:
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _JSON_FENCE_RE.sub("", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return value


def _raw_records(payload: dict) -> list:
    for key in ("records", "data", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    if "text" in payload and "metadata" in payload:
        return [payload]
    raise ValueError("JSON 中没有 records 数组")


def _flatten_path(value) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split("/") if part.strip()]
    if not isinstance(value, list):
        return []
    flat: list[str] = []
    for item in value:
        if isinstance(item, list):
            flat.extend(str(part).strip() for part in item if str(part).strip())
        elif str(item).strip():
            flat.append(str(item).strip())
    cleaned: list[str] = []
    for item in flat:
        title = clean_section_title(item)
        if title and (not cleaned or cleaned[-1] != title):
            cleaned.append(title)
    return cleaned


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _fill_labels(domain: str, subdomain: str, english: bool) -> tuple[str, str]:
    fallback = ("General", "General") if english else ("综合", "综合")
    return domain or fallback[0], subdomain or fallback[1]


def _labels_rejected(meta: dict[str, str]) -> bool:
    if (meta["domain"], meta["subdomain"]) in {("综合", "综合"), ("General", "General")}:
        return True
    return meta["language"] == "English" and (_has_cjk(meta["domain"]) or _has_cjk(meta["subdomain"]))


def _normalize_language(value: str, sample_text: str) -> str:
    chinese_parts = [
        part
        for part in re.split(r"\n\n+", sample_text)
        if part.strip() and not is_mostly_english(part)
    ]
    chinese = "\n".join(chinese_parts)
    cjk_kept = sum(1 for char in chinese if "\u4e00" <= char <= "\u9fff")
    judged = chinese if cjk_kept >= 40 else sample_text
    if document_is_english(judged):
        return "English"
    cjk = sum(1 for char in judged if "\u4e00" <= char <= "\u9fff")
    latin = sum(1 for char in judged if char.isascii() and char.isalpha())
    if cjk >= 40 and cjk > latin:
        return "中文"
    folded = value.strip().lower()
    if folded in {"中文", "zh", "chinese", "cn", "zh-cn"}:
        return "中文"
    if folded in {"英文", "en", "english"}:
        return "English"
    return "中文" if cjk >= latin else "English"


def coerce_records(raw_records: list, min_chars: int) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            continue
        text = cleanup_record_text(text)
        if len(text) < min_chars:
            continue
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        path = _flatten_path(metadata.get("section_path") or raw.get("section_path") or [])
        if not path:
            path = ["正文"]
        if is_dropped_section(path):
            continue
        key = (tuple(path), text)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "paper_title": str(raw.get("paper_title") or "").strip(),
                "text": text,
                "metadata": {
                    "domain": str(metadata.get("domain") or "").strip(),
                    "subdomain": str(metadata.get("subdomain") or "").strip(),
                    "language": str(metadata.get("language") or "").strip(),
                    "section_path": path,
                },
            }
        )
    return records


def _majority(values: list[str], fallback: str) -> str:
    counted = [value for value in values if value]
    if not counted:
        return fallback
    return Counter(counted).most_common(1)[0][0]


def apply_document_metadata(records: list[dict], meta: dict[str, str]) -> list[dict]:
    kept: list[dict] = []
    for record in records:
        path = list(record["metadata"]["section_path"])
        if path and path[0] == meta["paper_title"]:
            path = path[1:] or ["正文"]
        if is_dropped_section(path):
            continue
        record["paper_title"] = meta["paper_title"]
        record["metadata"]["domain"] = meta["domain"]
        record["metadata"]["subdomain"] = meta["subdomain"]
        record["metadata"]["language"] = meta["language"]
        record["metadata"]["section_path"] = path
        kept.append(record)
    return kept


def metadata_from_records(records: list[dict], title_hint: str) -> dict[str, str]:
    sample = "\n".join(record["text"][:400] for record in records[:3])
    titles = [record["paper_title"] for record in records]
    title = _majority(titles, title_hint)
    if title_hint and (not title or len(title) < 4 or len(title) > 80):
        title = title_hint
    pairs = [
        (record["metadata"]["domain"], record["metadata"]["subdomain"])
        for record in records
        if record["metadata"]["domain"] and record["metadata"]["subdomain"]
    ]
    language = _normalize_language(
        _majority([record["metadata"]["language"] for record in records], ""),
        sample,
    )
    if pairs:
        domain, subdomain = Counter(pairs).most_common(1)[0][0]
    else:
        domain, subdomain = _fill_labels("", "", language == "English")
    return {
        "paper_title": title or title_hint or "未命名",
        "domain": domain,
        "subdomain": subdomain,
        "language": language,
    }


_FOREIGN_PAGE_RE = re.compile(r"驾驶员|驾校|交通安全|珍惜生命")
_BROKEN_STITCH_RE = re.compile(r"[\u4e00-\u9fff]{1,4}\n+\s*[\u4e00-\u9fff]{1,2}为\d")


def _foreign_hits(text: str) -> int:
    return len(_FOREIGN_PAGE_RE.findall(text))


def _broken_stitch(text: str) -> str:
    match = _BROKEN_STITCH_RE.search(text)
    return match.group(0).replace("\n", "") if match else ""


def _page_penalty(text: str) -> int:
    penalty = _foreign_hits(text) * 10
    if _broken_stitch(text):
        penalty += 8
    if any(marker in text for marker in ("上接第", "下转第", "转下页", "接上页", "接下页")):
        penalty += 8
    if "珍惜生命" in text or "可塑性" in text:
        penalty += 6
    return penalty


def _page_markdown_dirty(title: str, text: str) -> bool:
    if any(marker in text for marker in ("上接第", "下转第", "转下页", "接上页", "接下页")):
        return True
    if _broken_stitch(text):
        return True
    return "交通" not in title and _foreign_hits(text) > 0


class PaperProcessor:
    def __init__(self, config: Config, client: LLMClient) -> None:
        self.config = config
        self.client = client

    def process_file(self, markdown_path: Path) -> list[dict]:
        raw = markdown_path.read_text(encoding="utf-8-sig", errors="replace")
        raw = self._reparse_continued_pdf(markdown_path, raw)
        trimmed = detach_extra_articles(raw)
        if trimmed != raw:
            logger.info("已去掉同文件中的其他论文，保留第一篇：%s", markdown_path.name)
            raw = trimmed
        repaired = detach_page_inserts(raw)
        if repaired != raw:
            logger.info("已剔除串入的其他文章并重接正文：%s", markdown_path.name)
            raw = repaired
        skip_reason = paper_skip_reason(raw)
        if skip_reason:
            logger.info("源文质量不足，跳过：%s (%s)", markdown_path.name, skip_reason)
            return []
        cleaned, title_hint = prepare_markdown(raw)
        if not cleaned:
            return []
        source = str(markdown_path.relative_to(self.config.input_dir))
        sections = _local_sections(cleaned)
        if not sections:
            return []
        fixes = [
            (section_index, piece_index, text)
            for section_index, (_, pieces) in enumerate(sections)
            for piece_index, (needs_fix, text) in enumerate(pieces)
            if needs_fix
        ]
        meta, replaced = self._classify_and_repair(cleaned, source, title_hint, fixes)
        records: list[dict] = []
        carried = ""
        for section_index, (path, pieces) in enumerate(sections):
            parts = [
                replaced.get((section_index, piece_index), text)
                for piece_index, (_, text) in enumerate(pieces)
            ]
            text = cleanup_record_text("\n\n".join(parts))
            if is_dropped_section(path):
                continue
            if carried:
                text = cleanup_record_text(f"{carried}\n\n{text}")
                carried = ""
            if len(text) < self.config.min_chars:
                carried = text
                continue
            records.append(
                {
                    "paper_title": meta["paper_title"],
                    "text": text,
                    "_source": "\n\n".join(piece for _, piece in pieces),
                    "metadata": {
                        "domain": meta["domain"],
                        "subdomain": meta["subdomain"],
                        "language": meta["language"],
                        "section_path": path or ["正文"],
                    },
                }
            )
        if carried and records:
            records[-1]["text"] = cleanup_record_text(f"{records[-1]['text']}\n\n{carried}")
        records = apply_document_metadata(records, meta)
        return _drop_low_quality_records(records, source, self.config.min_chars)

    def _reparse_continued_pdf(self, markdown_path: Path, raw: str) -> str:
        if not self.config.read_pdf_pages or not has_page_continuation(raw):
            return raw
        pdf = locate_pdf(markdown_path, self.config.input_dir, self.config.pdf_dir)
        if pdf is None:
            logger.info("有转页标记但未找到原 PDF：%s", markdown_path.name)
            return raw
        pages = render_pdf_pages(pdf, self.config)
        if not pages:
            return raw
        title_match = re.search(r"^#\s+(.+)$", raw, re.M)
        title = title_match.group(1).strip() if title_match else markdown_path.stem
        rebuilt = self._read_pdf_pages(pages, title)
        if not rebuilt:
            return raw
        logger.info("已按 PDF 页面图重读转页正文：%s（%s页）", markdown_path.name, len(pages))
        return rebuilt

    def _read_pdf_pages(self, pages: list[Path], title: str) -> str | None:
        markdown = self._ask_pages(pages, title, "")
        if not markdown:
            return None
        if _page_markdown_dirty(title, markdown):
            logger.info("页面图阅读仍夹有另一篇文章，再读一次")
            note = (
                "上一次把另一篇文章留在了正文里，或把后一页顶部无关文章的字粘进了本题。"
                "请删掉与题目无关的续文。下转前的最后一个字必须直接连上后一页中本题续文，合成一句。"
            )
            stitch = _broken_stitch(markdown)
            if stitch:
                note += f"这里接错了，请按原页改正：「{stitch}」。"
            retried = self._ask_pages(pages, title, note)
            if retried and _page_penalty(retried) < _page_penalty(markdown):
                markdown = retried
        markdown = tidy_page_markdown(title, markdown)
        return self._repair_dangling_sentence(pages, title, markdown)

    def _repair_dangling_sentence(self, pages: list[Path], title: str, markdown: str) -> str:
        match = re.search(r"(?:^|\n\n)(为\s*\d[^。\n]{0,80}。)", markdown)
        if not match:
            return markdown
        fragment = match.group(1)
        try:
            content = self.client.complete(
                build_stitch_messages(title, pages, fragment),
                min(512, self.config.max_output_tokens),
            )
            payload = _parse_json_object(content)
        except (LLMError, json.JSONDecodeError, ValueError, TruncatedError) as exc:
            logger.warning("转页断句补全失败：%s", exc)
            return markdown
        sentence = payload.get("sentence") or ""
        if not isinstance(sentence, str):
            return markdown
        sentence = sentence.strip()
        if len(sentence) < 12 or "为" not in sentence:
            return markdown
        if re.search(r"驾驶员|驾校|珍惜生命|可塑性|生命的脆弱|驾驶|性格", sentence):
            return markdown
        logger.info("已补全转页断句")
        return markdown.replace(fragment, sentence, 1)

    def _ask_pages(self, pages: list[Path], title: str, reject_note: str) -> str | None:
        try:
            content = self.client.complete(
                build_page_messages(title, pages, reject_note),
                min(16384, self.config.max_output_tokens),
            )
            payload = _parse_json_object(content)
        except (LLMError, json.JSONDecodeError, ValueError, TruncatedError) as exc:
            logger.warning("页面图阅读失败，沿用原 Markdown：%s", exc)
            return None
        markdown = payload.get("markdown") or payload.get("text") or ""
        if not isinstance(markdown, str) or len(markdown.strip()) < 80:
            logger.warning("页面图阅读结果过短，沿用原 Markdown")
            return None
        anchor = re.sub(r"\s+", "", title)[:8]
        if anchor and anchor not in re.sub(r"\s+", "", markdown):
            logger.warning("页面图阅读没有对上论文题目，沿用原 Markdown")
            return None
        return markdown.strip()

    def _classify_and_repair(
        self,
        cleaned: str,
        source: str,
        title_hint: str,
        fixes: list[tuple[int, int, str]],
    ) -> tuple[dict[str, str], dict[tuple[int, int], str]]:
        head = cleaned[:1800]
        if not fixes:
            return self._classify(cleaned, source, title_hint), {}
        english = document_is_english(cleaned)
        try:
            payload, replaced = self._repair_batch(fixes, head, source, title_hint, english)
        except (LLMError, json.JSONDecodeError, ValueError, TruncatedError) as exc:
            logger.warning("段落校对失败，保留本地正文：%s (%s)", source, exc)
            return self._classify(cleaned, source, title_hint), {}
        meta = self._meta_from_payload(payload, title_hint, cleaned)
        if _labels_rejected(meta):
            meta = self._classify(cleaned, source, title_hint)
        return meta, replaced

    def _repair_batch(
        self,
        fixes: list[tuple[int, int, str]],
        head: str,
        source: str,
        title_hint: str,
        english: bool,
    ) -> tuple[dict, dict[tuple[int, int], str]]:
        paragraphs = [(index, text) for index, (_, _, text) in enumerate(fixes)]
        dirty_chars = sum(len(text) for _, text in paragraphs)
        # 输出长度按原文字数从宽估计，避免生成到上限后整段重来。
        max_tokens = min(self.config.max_output_tokens, 800 + int(dirty_chars * 2.2))
        messages = build_fast_messages(head, source, title_hint, paragraphs, english)
        try:
            content = self.client.complete(messages, max_tokens)
            payload = _parse_json_object(content)
        except TruncatedError:
            if len(fixes) <= 1:
                raise
            middle = len(fixes) // 2
            logger.info("校对输出超长，拆成两段：%s", source)
            first_payload, first_map = self._repair_batch(
                fixes[:middle], head, source, title_hint, english
            )
            _, second_map = self._repair_batch(fixes[middle:], head, source, title_hint, english)
            first_map.update(second_map)
            return first_payload, first_map
        return payload, _replacement_map(payload, fixes, self.config.min_chars)

    def _meta_from_payload(self, payload: dict, title_hint: str, head: str) -> dict[str, str]:
        title = str(payload.get("paper_title") or "").strip()
        if title_hint and (not title or len(title) < 4 or len(title) > 80):
            title = title_hint
        language = _normalize_language(str(payload.get("language") or ""), head)
        domain, subdomain = _fill_labels(
            str(payload.get("domain") or "").strip(),
            str(payload.get("subdomain") or "").strip(),
            language == "English",
        )
        return {
            "paper_title": title or title_hint or "未命名",
            "domain": domain,
            "subdomain": subdomain,
            "language": language,
        }

    def describe(self, markdown_path: Path) -> str:
        raw = markdown_path.read_text(encoding="utf-8-sig", errors="replace")
        cleaned, title = prepare_markdown(raw)
        sections = [
            (path, block)
            for path, block in split_sections(cleaned)
            if not is_dropped_section(path)
        ]
        chunks = _pack_sections(sections, self.config.chunk_chars)
        mode = "single" if len(cleaned) <= self.config.single_pass_chars else "chunk"
        return (
            f"{markdown_path.name} chars={len(cleaned)} title={title or '-'} "
            f"sections={len(sections)} chunks={len(chunks)} mode={mode}"
        )

    def _generate_records(self, messages: list[dict[str, str]], char_len: int) -> list[dict]:
        last_error: Exception | None = None
        max_tokens = estimate_max_tokens(char_len, self.config.max_output_tokens)
        for attempt in range(1, self.config.parse_retries + 1):
            content = self.client.complete(messages, max_tokens)
            try:
                payload = _parse_json_object(content)
                return coerce_records(_raw_records(payload), self.config.min_chars)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "JSON 解析失败 %s/%s：%s；输出片段：%s",
                    attempt,
                    self.config.parse_retries,
                    exc,
                    content[:180].replace("\n", " "),
                )
        raise LLMError(f"无法解析模型输出：{last_error}")

    def _process_in_chunks(self, cleaned: str, source: str, title_hint: str) -> list[dict]:
        meta = self._classify(cleaned, source, title_hint)
        sections = [
            (path, block)
            for path, block in split_sections(cleaned)
            if not is_dropped_section(path)
        ]
        sections = _explode_large_sections(sections, self.config.chunk_chars)
        chunks = _pack_sections(sections, self.config.chunk_chars)
        records: list[dict] = []
        for index, chunk in enumerate(chunks, start=1):
            produced = self._generate_chunk(chunk, source, meta)
            records.extend(produced)
            logger.info("分块 %s/%s 得到 %s 条：%s", index, len(chunks), len(produced), source)
        return apply_document_metadata(records, meta)

    def _classify(self, cleaned: str, source: str, title_hint: str) -> dict[str, str]:
        head = cleaned[:3500]
        english = document_is_english(cleaned)
        messages = build_classify_messages(head, source, title_hint, english)
        try:
            content = self.client.complete(messages, 512)
            payload = _parse_json_object(content)
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("领域分类失败，使用候选题目：%s (%s)", source, exc)
            payload = {}
        title = str(payload.get("paper_title") or "").strip()
        if title_hint and (not title or len(title) < 4 or len(title) > 80):
            title = title_hint
        language = _normalize_language(str(payload.get("language") or ""), cleaned)
        domain, subdomain = _fill_labels(
            str(payload.get("domain") or "").strip(),
            str(payload.get("subdomain") or "").strip(),
            language == "English",
        )
        if language == "English" and (_has_cjk(domain) or _has_cjk(subdomain)):
            domain, subdomain = self._english_labels(domain, subdomain, head, source)
        return {
            "paper_title": title or title_hint or "未命名",
            "domain": domain,
            "subdomain": subdomain,
            "language": language,
        }

    def _english_labels(self, domain: str, subdomain: str, head: str, source: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": "你只输出 JSON 对象，不要解释。"},
            {
                "role": "user",
                "content": (
                    "下面的领域标签要改成简短英文专业词，不要保留汉字。"
                    '只输出 {"domain":"","subdomain":""}。\n'
                    f"domain：{domain}\nsubdomain：{subdomain}\n\n论文开头：\n{head[:1500]}"
                ),
            },
        ]
        try:
            content = self.client.complete(messages, 256)
            payload = _parse_json_object(content)
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("英文领域标签改写失败：%s (%s)", source, exc)
            return "General", "General"
        domain = str(payload.get("domain") or "").strip()
        subdomain = str(payload.get("subdomain") or "").strip()
        if not domain or not subdomain or _has_cjk(domain) or _has_cjk(subdomain):
            return "General", "General"
        return domain, subdomain

    def _generate_chunk(
        self,
        chunk: list[tuple[list[str], str]],
        source: str,
        meta: dict[str, str],
        depth: int = 0,
    ) -> list[dict]:
        char_len = sum(len(block) for _, block in chunk)
        messages = build_chunk_messages(chunk, source, meta)
        try:
            records = self._generate_records(messages, char_len)
            align_section_paths(records, [path for path, _ in chunk])
            return records
        except TruncatedError:
            if depth >= 3:
                raise
            smaller = _split_chunk(chunk)
            if len(smaller) <= 1 and smaller == [chunk]:
                raise
            records: list[dict] = []
            for part in smaller:
                records.extend(self._generate_chunk(part, source, meta, depth + 1))
            return records


def _local_sections(cleaned: str) -> list[tuple[list[str], list[tuple[bool, str]]]]:
    english_doc = document_is_english(cleaned)
    sections: list[tuple[list[str], list[tuple[bool, str]]]] = []
    for path, block in split_sections(cleaned):
        if is_dropped_section(path):
            continue
        lines = block.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if not path:
            body = clean_preamble(body, keep_english=english_doc)
            path = ["Abstract"] if english_doc else ["摘要"]
        if not body or (not english_doc and is_mostly_english(body)):
            continue
        pieces = section_paragraphs(body)
        if pieces:
            sections.append((path, pieces))
    return sections


def _replacement_map(
    payload: dict,
    fixes: list[tuple[int, int, str]],
    min_chars: int,
) -> dict[tuple[int, int], str]:
    items = payload.get("paragraphs") if isinstance(payload.get("paragraphs"), list) else []
    by_id: dict[int, str] = {}
    ordered: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            ordered.append(text)
            if "id" in item:
                try:
                    by_id[int(item["id"])] = text
                except (TypeError, ValueError):
                    pass
        elif isinstance(item, str):
            ordered.append(item)
    replaced: dict[tuple[int, int], str] = {}
    for index, (section_index, piece_index, original) in enumerate(fixes):
        updated = by_id.get(index)
        if updated is None and len(ordered) == len(fixes):
            updated = ordered[index]
        if not updated:
            continue
        updated = cleanup_record_text(updated)
        if len(updated) < max(min_chars, int(len(original) * 0.55)):
            continue
        replaced[(section_index, piece_index)] = updated
    return replaced


def align_section_paths(records: list[dict], hints: list[list[str]]) -> None:
    """模型若只返回了末级标题，用切分时的完整层级补回。"""
    usable = [hint for hint in hints if hint]
    for record in records:
        current = record["metadata"]["section_path"]
        for hint in usable:
            if current == hint or (len(current) > len(hint) and current[: len(hint)] == hint):
                break
            if len(current) < len(hint) and hint[-len(current) :] == current:
                record["metadata"]["section_path"] = list(hint)
                break


def _explode_large_sections(
    sections: list[tuple[list[str], str]],
    limit: int,
) -> list[tuple[list[str], str]]:
    exploded: list[tuple[list[str], str]] = []
    for path, block in sections:
        if len(block) <= limit:
            exploded.append((path, block))
            continue
        lines = block.splitlines()
        heading = lines[0] if lines and lines[0].lstrip().startswith("#") else ""
        body = "\n".join(lines[1:] if heading else lines)
        parts = split_paragraph_groups(body, max(1000, limit - len(heading) - 2)) or [body]
        for part in parts:
            text = f"{heading}\n\n{part}".strip() if heading else part.strip()
            if text:
                exploded.append((path, text))
    return exploded


def _pack_sections(
    sections: list[tuple[list[str], str]],
    limit: int,
) -> list[list[tuple[list[str], str]]]:
    chunks: list[list[tuple[list[str], str]]] = []
    current: list[tuple[list[str], str]] = []
    size = 0
    for path, block in sections:
        if current and size + len(block) > limit:
            chunks.append(current)
            current = []
            size = 0
        current.append((path, block))
        size += len(block)
    if current:
        chunks.append(current)
    return chunks


def _split_chunk(
    chunk: list[tuple[list[str], str]],
) -> list[list[tuple[list[str], str]]]:
    if len(chunk) > 1:
        middle = len(chunk) // 2
        return [chunk[:middle], chunk[middle:]]
    path, block = chunk[0]
    lines = block.splitlines()
    heading = lines[0] if lines and lines[0].lstrip().startswith("#") else ""
    body = "\n".join(lines[1:] if heading else lines)
    parts = split_paragraph_groups(body, max(1000, len(body) // 2))
    if len(parts) < 2:
        return [chunk]
    return [
        [(path, f"{heading}\n\n{part}".strip() if heading else part)]
        for part in parts
    ]


def write_paper_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_failure(config: Config, markdown_path: Path, error: str, lock: threading.Lock) -> None:
    payload = {"path": str(markdown_path), "error": error}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with lock:
        config.failed_path.parent.mkdir(parents=True, exist_ok=True)
        with config.failed_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def valid_record(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("paper_title"), str) or not record["paper_title"].strip():
        return False
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    for key in ("domain", "subdomain", "language"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            return False
    path = metadata.get("section_path")
    return (
        isinstance(path, list)
        and bool(path)
        and all(isinstance(item, str) and item.strip() for item in path)
    )


def _drop_low_quality_records(records: list[dict], source: str, min_chars: int = 40) -> list[dict]:
    kept: list[dict] = []
    rejected = 0
    for record in records:
        origin = record.pop("_source", None)
        reason = record_reject_reason(record)
        repaired = salvage_record(record, origin if isinstance(origin, str) else None, min_chars)
        section = record.get("metadata", {}).get("section_path")
        if repaired is None:
            rejected += 1
            logger.info("丢弃低质量片段：%s %s (%s)", source, section, reason or "无法根据原文修复")
            continue
        if reason:
            logger.info("已按原文修复片段：%s %s (%s)", source, section, reason)
        kept.append(repaired)
    if rejected >= 2 and rejected >= len(kept):
        logger.info("问题片段不少于保留片段，整篇不输出：%s", source)
        return []
    return kept


JSONL_SHARD_SIZE = 20000


def jsonl_output_paths(config: Config, paper_count: int) -> list[Path]:
    """输入不超过 20000 篇时返回单个 JSONL；超过则每 20000 篇一个文件。"""
    if paper_count <= JSONL_SHARD_SIZE:
        return [config.jsonl_path]
    stem = config.jsonl_path.stem
    suffix = config.jsonl_path.suffix or ".jsonl"
    shard_count = (paper_count + JSONL_SHARD_SIZE - 1) // JSONL_SHARD_SIZE
    width = max(5, len(str(shard_count)))
    return [
        config.jsonl_path.parent / f"{stem}_{index:0{width}d}{suffix}"
        for index in range(1, shard_count + 1)
    ]


def merge_jsonl(config: Config) -> tuple[int, int]:
    papers = discover_markdowns(config)
    destinations = jsonl_output_paths(config, len(papers))
    config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("w", encoding="utf-8") for path in destinations]
    file_count = 0
    record_count = 0
    try:
        for index, markdown_path in enumerate(papers):
            json_path = output_json_path(config, markdown_path)
            if not json_path.is_file():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("跳过无法解析的 JSON：%s", json_path)
                continue
            if not isinstance(data, list):
                logger.warning("跳过非数组 JSON：%s", json_path)
                continue
            file_count += 1
            handle = handles[index // JSONL_SHARD_SIZE]
            for record in data:
                if not valid_record(record):
                    logger.warning("跳过字段不完整的记录：%s", json_path)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1
    finally:
        for handle in handles:
            handle.close()
    logger.info(
        "已汇总 %s 个 JSON，%s 条样本，共 %s 个 JSONL -> %s",
        file_count,
        record_count,
        len(destinations),
        ", ".join(path.name for path in destinations),
    )
    for path in destinations:
        audit_jsonl(path)
    return file_count, record_count
