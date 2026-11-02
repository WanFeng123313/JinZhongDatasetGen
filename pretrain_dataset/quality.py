"""宁缺毋滥：识别串页、坏公式、异常数值和表格错位。先按原文修复，修不好再丢弃。"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from pretrain_dataset.text_norm import cleanup_record_text, strip_furniture_lines, strip_reference_paragraphs

logger = logging.getLogger("pretrain_dataset")

# 分页接排说明这一页和下一段不是连续正文。
_PAGE_RE = re.compile(
    r"(?:上接|下转|转下页|接下页|接上页|下接第|上接第).{0,8}\d+\s*页"
    r"|continued\s+(?:on|from)\s+page\s+\d+"
    r"|未完待续",
    re.I,
)
_CTRL_RE = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f"
    r"\u200b-\u200f\u202a-\u202e\u2060\ufeff\ufffd]"
)
# 三位及以上的整数百分数，如 738%、1000％。200% 以内仍可能是真实增幅。
_PCT_HI_RE = re.compile(r"(?<![\d.\\])(\d{3,})\s*(?<!\\)[%％]")
# “51 9%”“96 70%”这类被拆开的百分数，小数点已经丢失。
_PCT_GAP_RE = re.compile(r"\d+\s+\d+\s*(?<!\\)[%％]")
# “07%”“07％”多半是 0.7% 丢掉了小数点。
_PCT_ZERO_RE = re.compile(r"(?<![\d.\\])0\d{1,2}\s*(?<!\\)[%％]")
# mg/L 被识别成 m/L。前面是字母时不匹配，避免误伤 mg/L、km/L。
_BARE_ML_RE = re.compile(r"(?<![A-Za-zμµ])m\s*[/／]\s*L")
_BROKEN_UNIT_RE = re.compile(
    r"亿\s*[²³²³]|用\s*[²³²³]|(?<![A-Za-zμµ])k\s+²|(?<![mMkK])k²"
)
_GARBAGE_MATH_RE = re.compile(
    r"\\nparallel|\\sharp\b|\\odot\b|[☉⊕⊗※]"
    r"|\\#(?:\s*\\#){1,}"
    r"|\\mathrm\s*\{\s*\{?\s*(?:[A-Za-z]\s+){3,}[A-Za-z]"
)
_REF_LINE_RE = re.compile(r"^[［\[]\s*\d+\s*[］\]]", re.M)
_JOURNAL_RE = re.compile(r"［J］|\[J\]")
_FRONT_ABS_RE = re.compile(r"摘要\s*[:：]")
_FRONT_KW_RE = re.compile(r"关键[词字]")
_FRONT_ID_RE = re.compile(r"中图分类号|文章编号|文献标识码")
# 标题里没有“交通”时，密集出现这些词说明串入了另一篇文章。
_TRAFFIC_RE = re.compile(r"驾驶员|驾校|交通事故|交通安全|交通规则")


_MARKER_RE = re.compile(
    r"[（(]?\s*(?P<kind>下转|转下页|上接|接上页|接下页|下接|接上)\s*第?\s*\d+\s*页\s*[）)]?"
    r"|continued\s+(?P<en>on|from)\s+page\s+\d+",
    re.I,
)
_STOP_GRAM = set("的了是在和与及或而并对把被从到为以就也都还这那其中上下里")
_GENERIC_GRAM = (
    "技术",
    "研究",
    "方法",
    "分析",
    "问题",
    "系统",
    "工程",
    "发展",
    "建设",
    "管理",
    "应用",
    "影响",
    "作用",
    "模型",
    "数据",
    "实验",
    "试验",
    "结果",
)


def _marker_kind(match: re.Match[str]) -> str:
    kind = match.group("kind")
    if kind:
        return "down" if kind in {"下转", "转下页", "下接"} else "up"
    return "down" if match.group("en").lower() == "on" else "up"


def _content_grams(text: str) -> set[str]:
    chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    grams = {
        chars[index : index + 3]
        for index in range(len(chars) - 2)
        if not (set(chars[index : index + 3]) & _STOP_GRAM)
        and not any(word in chars[index : index + 3] for word in _GENERIC_GRAM)
    }
    grams.update(token.lower() for token in re.findall(r"[A-Za-z]{5,}", text))
    return grams


def _first_heading(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.M)
    return match.group(1) if match else ""


def _stitch(left: str, right: str, hard_break: bool) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if hard_break or right.startswith("#") or left[-1] in "。！？；：\n":
        return left + "\n\n" + right
    return left + right


def _cut_page_inserts(text: str) -> str:
    """去掉下转到上接之间的他文，并接上被截断的句子。单独的上接改成段落边界。"""
    marks = list(_MARKER_RE.finditer(text))
    if not marks:
        return text
    merged = ""
    cursor = 0
    index = 0
    soft = False
    while index < len(marks):
        mark = marks[index]
        merged = _stitch(merged, text[cursor : mark.start()], hard_break=bool(merged) and not soft)
        if _marker_kind(mark) == "down":
            follow = index + 1
            while follow < len(marks) and _marker_kind(marks[follow]) != "up":
                follow += 1
            if follow < len(marks):
                cursor = marks[follow].end()
                index = follow + 1
                soft = True
                continue
            cursor = mark.end()
            index += 1
            soft = False
        else:
            cursor = mark.end()
            index += 1
            soft = False
    return _stitch(merged, text[cursor:], hard_break=bool(merged) and not soft)


def _host_profile(text: str) -> set[str]:
    heading = _first_heading(text)
    first_mark = _MARKER_RE.search(text)
    opening = text[: first_mark.start()] if first_mark else text[:4000]
    return _content_grams(heading + "\n" + opening)


def _keep_host_block(paragraph: str, profile: set[str], protect: bool) -> bool:
    if protect:
        return True
    stripped = paragraph.strip()
    if not stripped:
        return False
    if stripped.startswith("|") or "$$" in stripped:
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    latin = len(re.findall(r"[A-Za-z]", stripped))
    if stripped.startswith("#"):
        return len(_content_grams(stripped) & profile) >= 1
    if cjk < 40 and latin < 80:
        return True
    if not profile:
        return True
    return len(_content_grams(stripped) & profile) >= 2


_LEAK_SENTENCE_RE = re.compile(
    r"驾驶员|驾校|交通安全|珍惜生命|可塑性|生命的脆弱|它的珍惜|驾驶|性格|领悟到"
)


def tidy_page_markdown(title: str, text: str) -> str:
    """页面图把后一页顶部的另一篇文章夹进正文时，丢掉夹段并把断句接上。"""
    text = re.sub(r"(最低标)高?\s*\n+\s*性?\s*(为\s*\d)", r"\1高\2", text)
    if "交通" in title:
        return text
    profile = _content_grams(title + "\n" + text[:1500])
    paragraphs = [part.strip() for part in re.split(r"\n\n+", text.strip()) if part.strip()]
    kept: list[str] = []
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        if paragraph.endswith("最低标高"):
            follow = index + 1
            while follow < len(paragraphs) and not paragraphs[follow].startswith("为"):
                follow += 1
            if follow < len(paragraphs):
                kept.append(paragraph + paragraphs[follow])
                index = follow + 1
                continue
        if paragraph.startswith("|") or paragraph.startswith("#") or "$$" in paragraph:
            kept.append(paragraph)
            index += 1
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[。！？])", paragraph) if part.strip()]
        host = [
            sentence
            for sentence in (sentences or [paragraph])
            if sentence and not _LEAK_SENTENCE_RE.search(sentence)
        ]
        block = "".join(host).strip()
        if block and (_keep_host_block(block, profile, False) or block.startswith("为")):
            kept.append(block)
        index += 1
    return "\n\n".join(kept)


def detach_page_inserts(text: str) -> str:
    """分页接排时丢掉不属于本文的文字，再把本文被截断的句子接回去。"""
    if len(_PAGE_RE.findall(text)) < 2 and len(list(_MARKER_RE.finditer(text))) < 2:
        return text
    profile = _host_profile(text)
    rewritten = _cut_page_inserts(text)
    kept: list[str] = []
    for paragraph in re.split(r"\n\n+", rewritten):
        if _keep_host_block(paragraph, profile, False):
            kept.append(paragraph.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(part for part in kept if part)).strip()


def detach_extra_articles(text: str) -> str:
    """同一 Markdown 里排进下一篇论文时，只保留第一篇。没有第二篇标题时只删重复的文章编号行。"""
    marks = list(re.finditer(r"文章编号", text))
    if len(marks) < 2:
        return text
    headings = list(re.finditer(r"^#\s+\S.+$", text[: marks[1].start()], re.M))
    if len(headings) < 2:
        return _drop_repeated_article_id_lines(text)
    start = headings[1].start()
    doi = list(re.finditer(r"^DOI[^\n]*", text[:start], re.M | re.I))
    if doi and start - doi[-1].start() < 500:
        start = doi[-1].start()
    return text[:start].rstrip()


def _drop_repeated_article_id_lines(text: str) -> str:
    seen = False
    kept: list[str] = []
    for line in text.splitlines():
        compact = re.sub(r"\s", "", line)
        if "文章编号" in line and len(compact) <= 48:
            if seen:
                continue
            seen = True
        kept.append(line)
    return "\n".join(kept).strip()


def paper_skip_reason(text: str) -> str | None:
    """源 Markdown 整篇都不可用时返回原因。分页接排先剔除他文，不因此整篇跳过。"""
    if len(re.findall(r"文章编号", text)) >= 2:
        return "源文件含有多篇论文题录"
    return None


def record_reject_reason(record: dict) -> str | None:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return "正文为空"
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    path = metadata.get("section_path") if isinstance(metadata.get("section_path"), list) else []
    path_text = "\n".join(str(part) for part in path)
    if _PAGE_RE.search(text) or _PAGE_RE.search(path_text):
        return "分页接排"
    if _CTRL_RE.search(text):
        return "含不可见控制字符"
    if any(int(value) >= 300 for value in _PCT_HI_RE.findall(text)):
        return "百分数异常"
    if _PCT_GAP_RE.search(text) or _PCT_ZERO_RE.search(text):
        return "百分数小数点丢失"
    if _BARE_ML_RE.search(text) or _BROKEN_UNIT_RE.search(text):
        return "单位损坏"
    if _GARBAGE_MATH_RE.search(text):
        return "公式识别损坏"
    if _table_is_broken(text):
        return "表格列错位"
    if _FRONT_ABS_RE.search(text) and _FRONT_KW_RE.search(text) and _FRONT_ID_RE.search(text):
        return "混入其他论文"
    if len(_REF_LINE_RE.findall(text)) >= 3 or len(_JOURNAL_RE.findall(text)) >= 2:
        return "混入参考文献"
    title = str(record.get("paper_title") or "")
    if "交通" not in title and "交通" not in path_text and len(_TRAFFIC_RE.findall(text)) >= 4:
        return "正文串入无关内容"
    return None


def _split_table_row(row: str) -> list[str]:
    text = row.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    cells: list[str] = []
    buffer: list[str] = []
    in_math = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "$":
            if index + 1 < len(text) and text[index + 1] == "$":
                in_math = not in_math
                buffer.append("$$")
                index += 2
                continue
            in_math = not in_math
            buffer.append(char)
        elif char == "|" and not in_math:
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        index += 1
    cells.append("".join(buffer).strip())
    return cells


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def _table_is_broken(text: str) -> bool:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line.strip())
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    for rows in groups:
        if len(rows) < 2:
            continue
        parsed = [_split_table_row(row) for row in rows]
        if any(re.search(r"^\|\s*\}", row) for row in rows):
            return True
        counts = [len(cells) for cells in parsed]
        if max(counts) - min(counts) >= 2:
            return True
        width = counts[0]
        if width < 4:
            continue
        content = [cells for cells in parsed if not _is_separator_row(cells)]
        seen_data = False
        for index, cells in enumerate(content):
            filled = [cell for cell in cells if cell]
            numeric = any(re.search(r"\d", cell) for cell in filled)
            # 表头和它下面的分组子表头出现在第一条数字行之前，不是列错位。
            if index == 0 or not seen_data:
                if numeric and index > 0:
                    seen_data = True
                continue
            if numeric or not filled or len(filled) / width > 0.5:
                if numeric:
                    seen_data = True
                continue
            if index + 1 >= len(content):
                continue
            nxt = [cell for cell in content[index + 1] if cell]
            if len(nxt) / width >= 0.8 and any(re.search(r"\d", cell) for cell in nxt):
                return True
    return False


def _cjk_text(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", text))


def _ordered_coverage(needle: str, haystack: str) -> float:
    if not needle:
        return 0.0
    index = 0
    for char in haystack:
        if index < len(needle) and char == needle[index]:
            index += 1
    return index / len(needle)


def _hard_span_defect(text: str) -> bool:
    if any(int(value) >= 300 for value in _PCT_HI_RE.findall(text)):
        return True
    if _PCT_GAP_RE.search(text) or _PCT_ZERO_RE.search(text):
        return True
    if _BARE_ML_RE.search(text) or _BROKEN_UNIT_RE.search(text):
        return True
    return _GARBAGE_MATH_RE.search(text) is not None


def _clean_section_path(path: list[str]) -> list[str]:
    cleaned: list[str] = []
    for part in path:
        text = _MARKER_RE.sub("", str(part))
        text = _PAGE_RE.sub("", text)
        text = re.sub(r"[（(]\s*[）)]", "", text).strip()
        if text:
            cleaned.append(text)
    return cleaned or ["正文"]


def _strip_page_markers(text: str) -> str:
    text = _MARKER_RE.sub("", text)
    text = _PAGE_RE.sub("", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    return text


def _strip_reference_lines(text: str) -> str:
    if len(_REF_LINE_RE.findall(text)) < 3 and len(_JOURNAL_RE.findall(text)) < 2:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _REF_LINE_RE.match(stripped):
            continue
        if _JOURNAL_RE.search(stripped) and "。" not in stripped:
            continue
        kept.append(line)
    return strip_reference_paragraphs("\n".join(kept))


def _drop_front_matter(text: str) -> str:
    if not (_FRONT_ABS_RE.search(text) and _FRONT_KW_RE.search(text) and _FRONT_ID_RE.search(text)):
        return text
    kept: list[str] = []
    for paragraph in re.split(r"\n\n+", text):
        if _FRONT_ID_RE.search(paragraph):
            continue
        if _FRONT_ABS_RE.search(paragraph) and _FRONT_KW_RE.search(paragraph):
            continue
        if re.match(r"\s*摘要\s*[:：]", paragraph) or re.match(r"\s*关键[词字]\s*[:：]", paragraph):
            continue
        if paragraph.strip():
            kept.append(paragraph.strip())
    return "\n\n".join(kept)


def _drop_traffic(text: str, title: str, path_text: str) -> str:
    if "交通" in title or "交通" in path_text or len(_TRAFFIC_RE.findall(text)) < 4:
        return text
    kept: list[str] = []
    for paragraph in re.split(r"\n\n+", text):
        if not _TRAFFIC_RE.search(paragraph):
            if paragraph.strip():
                kept.append(paragraph.strip())
            continue
        sentences = [part for part in re.split(r"(?<=[。！？])", paragraph) if part.strip()]
        host = [part.strip() for part in sentences if not _TRAFFIC_RE.search(part)]
        if host:
            kept.append("".join(host))
    return "\n\n".join(kept)


def _table_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _usable_source_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(_cjk_text(stripped)) < 6 and len(re.findall(r"[A-Za-z]", stripped)) < 24:
        return False
    if _PAGE_RE.search(stripped) or _MARKER_RE.search(stripped) or _hard_span_defect(stripped):
        return False
    if _table_is_broken(stripped):
        return False
    return True


def _source_sentences(source: str) -> list[str]:
    pieces: list[str] = []
    for paragraph in re.split(r"\n\n+", source):
        if paragraph.strip().startswith("|") or "$$" in paragraph:
            pieces.append(paragraph.strip())
            continue
        parts = [part.strip() for part in re.split(r"(?<=[。！？])", paragraph) if part.strip()]
        pieces.extend(parts or [paragraph.strip()])
    return pieces


def _best_source_span(bad: str, candidates: list[str]) -> str | None:
    needle = _cjk_text(bad)
    if len(needle) < 6:
        return None
    best: str | None = None
    best_score = 0.0
    for candidate in candidates:
        if not _usable_source_sentence(candidate):
            continue
        if len(candidate) > max(500, len(bad) * 4):
            continue
        score = _ordered_coverage(needle, _cjk_text(candidate))
        if score > best_score:
            best = candidate.strip()
            best_score = score
    if best is not None and best_score >= 0.75:
        return best
    return None


def _repair_tables(text: str, source: str) -> str:
    source_tables = [block for block in _table_blocks(source) if not _table_is_broken(block) and not _hard_span_defect(block)]
    for block in _table_blocks(text):
        if not _table_is_broken(block) and not _hard_span_defect(block):
            continue
        header = _cjk_text(" ".join(_split_table_row(block.splitlines()[0])))
        replacement = _best_source_span(header, source_tables) if len(header) >= 4 else None
        if replacement is None:
            for table in source_tables:
                source_header = _cjk_text(" ".join(_split_table_row(table.splitlines()[0])))
                if header and _ordered_coverage(header, source_header) >= 0.6:
                    replacement = table
                    break
        text = text.replace(block, replacement or "")
    return text


def _repair_prose(text: str, source: str) -> str:
    candidates = _source_sentences(source) if source else []
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\n+", text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if stripped.startswith("|") or (stripped.startswith("$$") and stripped.endswith("$$")):
            if not _hard_span_defect(stripped) and not _table_is_broken(stripped):
                paragraphs.append(stripped)
            continue
        if not _hard_span_defect(stripped):
            paragraphs.append(stripped)
            continue
        parts = [part.strip() for part in re.split(r"(?<=[。！？])", stripped) if part.strip()]
        if len(parts) <= 1:
            replaced = _best_source_span(stripped, candidates)
            if replaced:
                paragraphs.append(replaced)
            continue
        kept: list[str] = []
        for part in parts:
            if not _hard_span_defect(part):
                kept.append(part)
                continue
            replaced = _best_source_span(part, candidates)
            if replaced:
                kept.append(replaced)
        if kept:
            paragraphs.append("".join(kept))
    return "\n\n".join(paragraphs)


def _repair_text(text: str, source: str, title: str, path_text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = strip_furniture_lines(text)
    text = _strip_page_markers(text)
    text = _strip_reference_lines(text)
    text = _drop_front_matter(text)
    text = _drop_traffic(text, title, path_text)
    text = _repair_tables(text, source)
    text = _repair_prose(text, source)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def salvage_record(record: dict, source: str | None, min_chars: int = 40) -> dict | None:
    """先按 Markdown 原文修片段。修完仍不合格，或剩下的正文太短，就丢掉。"""
    if record_reject_reason(record) is None:
        return {key: value for key, value in record.items() if key != "_source"}
    metadata = dict(record.get("metadata") or {})
    path = metadata.get("section_path") if isinstance(metadata.get("section_path"), list) else []
    path = _clean_section_path([str(part) for part in path])
    title = str(record.get("paper_title") or "")
    path_text = "\n".join(path)
    raw_text = record.get("text") if isinstance(record.get("text"), str) else ""
    corpus = cleanup_record_text(source) if isinstance(source, str) and source.strip() else ""
    repaired = _repair_text(raw_text, corpus, title, path_text)
    candidate = {key: value for key, value in record.items() if key != "_source"}
    candidate["text"] = repaired
    candidate["metadata"] = {**metadata, "section_path": path}
    if record_reject_reason(candidate) is None and len(repaired) >= min_chars:
        return candidate
    if corpus:
        from_source = _repair_text(corpus, "", title, path_text)
        candidate["text"] = from_source
        if record_reject_reason(candidate) is None and len(from_source) >= min_chars:
            return candidate
    return None


def audit_jsonl(path: Path) -> tuple[int, int]:
    """逐行检查已生成的 JSONL。能按原文规则修好的保留，修不好的删除。"""
    if not path.is_file():
        logger.info("没有 JSONL 可检查：%s", path)
        return 0, 0
    parsed: list[tuple[dict, str | None, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                parsed.append(({}, "无法解析", raw))
                logger.warning("删除 JSONL 第%s行：无法解析", line_no)
                continue
            if not isinstance(record, dict):
                parsed.append(({}, "不是对象", raw))
                continue
            reason = record_reject_reason(record)
            repaired = salvage_record(record, None)
            if repaired is None:
                parsed.append((record, reason or "无法根据原文修复", raw))
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            repaired_meta = repaired.get("metadata") if isinstance(repaired.get("metadata"), dict) else {}
            if repaired.get("text") != record.get("text") or repaired_meta.get("section_path") != metadata.get("section_path"):
                raw = json.dumps(repaired, ensure_ascii=False)
            parsed.append((repaired, None, raw))

    kept: list[str] = []
    removed: Counter[str] = Counter()
    for _record, reason, raw in parsed:
        if reason:
            removed[reason] += 1
            continue
        kept.append(raw)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    temporary.replace(path)
    detail = "，".join(f"{name} {count}" for name, count in removed.most_common()) or "无"
    logger.info(
        "逐行检查完成：删除 %s 条，保留 %s 条（%s） -> %s",
        sum(removed.values()),
        len(kept),
        detail,
        path,
    )
    return len(kept), sum(removed.values())


def self_check() -> None:
    good = {
        "paper_title": "水库蒸发计算",
        "text": "水库多年平均蒸发量为12.4mm，出库流量合格率92%。损失量按 $W=E-R$ 计算，浓度约为1.3mg/L。",
        "metadata": {"section_path": ["蒸发损失"]},
    }
    assert record_reject_reason(good) is None
    captioned = {
        **good,
        "text": "由图3可知，1981～2010年桂阳烟区温差呈现缓慢减小的趋势，年平均值为7.46℃。\nFig. 3 Variation trend of temperature.",
    }
    assert record_reject_reason(captioned) is None
    assert record_reject_reason(
        {**good, "text": "矿坑充水有限。\n\n（上接第107页）\n\n大气降雨是主要充水水源。"}
    ) == "分页接排"
    assert record_reject_reason(
        {**good, "metadata": {"section_path": ["（上接第90页）"]}, "text": "后文继续讨论水质。"}
    ) == "分页接排"
    assert record_reject_reason({**good, "text": "工程规模以＜50t（738％）为主，合格率1000％。"}) == "百分数异常"
    assert record_reject_reason({**good, "text": "固定沙地占沙化土地的96 70%。"}) == "百分数小数点丢失"
    assert record_reject_reason({**good, "text": "出水TP平均为0.5m/L，进水为1.3mg/L。"}) == "单位损坏"
    assert record_reject_reason(
        {**good, "text": r"式中 $W _ { \mathrm { { f i f l } } \bar { \nparallel } }$ 为损失量。"}
    ) == "公式识别损坏"
    assert record_reject_reason({**good, "text": "正常公式 $$Q = C I A$$ 仍然保留。"}) is None
    broken_table = (
        "见表1。\n\n"
        "| 计算范围 | 汇水面积 | 径流系数 | 涌水量 |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 雨季日最大涌水量 | 平均涌水 |  |  |  |\n"
        "| 境界内 | 320830.93 | 0.8 | 9240 | 84.7 |"
    )
    assert record_reject_reason({**good, "text": broken_table}) is None
    shifted_table = (
        "见表1。\n\n"
        "| 范围 | 面积 | 系数 | 涌水量 |\n"
        "| --- | --- | --- | --- |\n"
        "| 境界内 | 320830.93 | 0.8 | 9240 |\n"
        "| 表头错位 | 平均涌水 |  |  |\n"
        "| 境界外 | 982919.83 | 0.2 | 7077 |"
    )
    assert record_reject_reason({**good, "text": shifted_table}) == "表格列错位"
    aligned = (
        "| 测次 | 1 | 2 | 3 |\n"
        "| --- | --- | --- | --- |\n"
        "| 测值 | 797 | 801 | 799 |\n"
        "| 重复性误差 |  | 0.25% |  |"
    )
    assert record_reject_reason({**good, "text": aligned}) is None
    grouped = (
        "|  | 零点漂移前 |  | 零点漂移后 |\n"
        "| --- | --- | --- | --- |\n"
        "|  | 1 | 2 | 3 |\n"
        "| 测量值 | 802 | 801 | 814 |"
    )
    assert record_reject_reason({**good, "text": grouped}) is None
    math_table = (
        r"| 指标 | 误差 |"
        "\n| --- | --- |\n"
        r"| 精度 | $\left\| \frac{f_1-f_0}{f_0} \right\|$ |"
    )
    assert record_reject_reason({**good, "text": math_table}) is None
    clash = (
        "驾驶员培训时要防止交通事故。交通安全与驾驶员性格有关，驾校应培养交通安全意识，"
        "驾驶员还必须遵守交通规则。\n\n取地表迳流系数经验值，境界内取0.8。"
    )
    assert record_reject_reason({**good, "text": clash}) == "正文串入无关内容"
    assert paper_skip_reason("正文（上接第12页）后文（下转第13页）结束") is None
    assert paper_skip_reason("文章编号：1\n正文\n文章编号：2") == "源文件含有多篇论文题录"
    two_papers = (
        "# 城市公园节水绿地设计研究\n\n摘 要 讨论城市公园如何利用雨水建设节水绿地。\n\n"
        "文章编号：1001-0009（2015）08-0101-06\n\n公园北部适宜作为雨水下渗区域。\n\n"
        "DOI：10.11937/bfyy.201508028\n\n# 园林植物配植设计立意的研究\n\n"
        "摘 要 讨论植物配植的立意方法。\n\n文章编号：1001-0009（2015）08-0106-03\n\n"
        "立意决定植物配植方案的走向。\n"
    )
    kept_paper = detach_extra_articles(two_papers)
    assert "节水绿地" in kept_paper and "园林植物" not in kept_paper
    assert paper_skip_reason(kept_paper) is None
    repeated = "# 题目\n\n文章编号：1001-0009\n\n正文讨论城市绿地的雨水利用，资料覆盖一个完整的公园。\n\n文章编号：1001-0009\n\n后文继续讨论下凹式绿地。\n"
    kept_repeat = detach_extra_articles(repeated)
    assert kept_repeat.count("文章编号") == 1 and "下凹式绿地" in kept_repeat
    spliced = """# 浅论湿地植被类型特征

兴安盟湿地植被以柴桦灌丛和苔草沼泽为主。

## 1.2 金老梅灌丛群系

高度约70cm左（下转第126页）

今后甘河林业局以野猪驯养为重点，全力发展野猪驯养基地，引进优良母猪并修建猪舍。

（上接第92页）右，常见伴生种有柴桦、柳叶绣线菊和灰脉苔草。

## 1.3 柳灌丛

主要在科尔沁保护区分布，伴生种有柴桦和苔草。
"""
    detached = detach_page_inserts(spliced)
    assert "野猪" not in detached
    assert "左右" in detached
    assert "柴桦" in detached
    inline = """# 畜牧生态模式

牧民采用草场改良和家庭养殖，草场面积和牧草产量都有提高。人工种植饲料面积(上接第6页)恢复正常的生态系统。活菌制剂用于仔猪消化道，防止仔猪腹泻。

此种模式减轻了畜牧业对草场的压力，草场植被得以恢复。
"""
    detached_inline = detach_page_inserts(
        inline.replace(
            "防止仔猪腹泻。",
            "防止仔猪腹泻，并提高哺乳仔猪抗体水平和饲料消化率，过量则会造成仔猪下痢。",
        )
        + "\n\n（下转第8页）\n\n另一处无关文字。"
    )
    assert "仔猪" not in detached_inline
    assert "草场" in detached_inline
    page_text = (
        "矿坑充水有限，大气降水是矿坑的主要充水水源，涌水量随降雨明显变化。"
        "\n\n（上接第107页）\n\n"
        "大气降雨是主要充水水源，旱季流量明显减小，雨季则迅速增大。"
    )
    saved = salvage_record({**good, "text": page_text}, None)
    assert saved is not None and "上接" not in saved["text"] and "旱季" in saved["text"]
    assert record_reject_reason(saved) is None
    path_saved = salvage_record(
        {
            **good,
            "metadata": {"section_path": ["（上接第90页）"]},
            "text": "后文继续讨论该河段的水质变化，旱季流量明显减小，雨季则迅速增大，多年平均矿化度保持稳定。",
        },
        None,
    )
    assert path_saved is not None and path_saved["metadata"]["section_path"] == ["正文"]
    mixed = (
        "桂阳烟区烤烟生长期温差呈现缓慢减小的趋势，年平均减小幅度为738%。"
        "邻近站点的日照时间变化不大，年平均增加幅度为0.56%。"
    )
    source_mixed = (
        "桂阳烟区烤烟生长期温差呈现缓慢减小的趋势，年平均减小幅度为0.16%。"
        "邻近站点的日照时间变化不大，年平均增加幅度为0.56%。"
    )
    restored = salvage_record({**good, "text": mixed}, source_mixed)
    assert restored is not None and "738" not in restored["text"] and "0.16%" in restored["text"]
    assert salvage_record(
        {**good, "text": "工程规模以＜50t（738％）为主，合格率1000％。"},
        "工程规模以＜50t（738％）为主，合格率1000％。",
    ) is None
    unit_saved = salvage_record(
        {**good, "text": "出水TP平均为0.5m/L，进水为1.3mg/L，去除效果稳定，可以满足回用要求。"},
        "出水TP平均为0.5mg/L，进水为1.3mg/L，去除效果稳定，可以满足回用要求。",
    )
    assert unit_saved is not None and "m/L" not in unit_saved["text"].replace("mg/L", "")
    assert "0.5mg/L" in unit_saved["text"]
    prose = "见表1，境界内雨季涌水量按汇水面积和径流系数估算，多年平均来水量约为9240立方米，可供设计选用。"
    table_saved = salvage_record({**good, "text": prose + "\n\n" + shifted_table}, None)
    assert table_saved is not None and "表头错位" not in table_saved["text"] and "9240" in table_saved["text"]
    formula = (
        "水库多年平均蒸发量为12.4mm，出库流量合格率92%。"
        r"式中 $W _ { \mathrm { { f i f l } } \bar { \nparallel } }$ 为损失量。"
        "浓度约为1.3mg/L。"
    )
    formula_saved = salvage_record({**good, "text": formula}, None)
    assert formula_saved is not None and "nparallel" not in formula_saved["text"] and "蒸发量" in formula_saved["text"]
    clash_saved = salvage_record(
        {**good, "text": clash + "多年平均径流量据此估算，可供该矿区排水设计选用。"},
        None,
    )
    assert clash_saved is not None and "驾校" not in clash_saved["text"] and "迳流" in clash_saved["text"]
    cited = (
        "本文分析了水库蒸发损失，多年平均蒸发量为12.4mm，出库流量合格率92%，浓度约为1.3mg/L。\n\n"
        "［1］ 张三. 论文［J］. 期刊，2010.\n"
        "［2］ 李四. 论文［J］. 期刊，2011.\n"
        "［3］ 王五. 论文［J］. 期刊，2012."
    )
    cited_saved = salvage_record({**good, "text": cited}, None)
    assert cited_saved is not None and "［1］" not in cited_saved["text"] and "蒸发" in cited_saved["text"]
    tidied = tidy_page_markdown(
        "巴基斯坦山达克铜矿北矿段水文地质条件分析",
        "矿区地下水的主要补给源是大气降水，大气降水通过风化裂隙渗入矿体。\n\n"
        "矿床露天开采最低标高\n\n"
        "应深深领悟到生命的脆弱与顽强，它取决于你对它的珍惜。\n\n"
        "为753m，区内及相邻地下水对矿坑充水无影响。",
    )
    assert "最低标高为753m" in tidied
    assert "生命的脆弱" not in tidied
    assert "大气降水" in tidied
