from __future__ import annotations

import html
import re

_CJK = r"\u4e00-\u9fff"
_NUM = r"0-9０-９"
_HS = "[ \t\u00a0\u3000]"

_MATH_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FIG_CAPTION_RE = re.compile(rf"^(图|圖|Fig\.?|Figure)\s*[{_NUM}]", re.I)
_CAPTION_SPACE_RE = re.compile(
    rf"^(表|图)([0-9０-９]+(?:[\.．][0-9０-９]+)*)(?=[{_CJK}])",
    re.M,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SUP_RE = re.compile(r"<sup[^>]*>([\s\S]*?)</sup>", re.I)
_SUB_RE = re.compile(r"<sub[^>]*>([\s\S]*?)</sub>", re.I)
_SUPER_DIGITS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUB_DIGITS = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_CITATION_LATEX_RE = re.compile(
    r"\$\s*\^\s*\{\s*[0-9０-９①-⑳，,、\s]+\s*\}\s*\$"
)
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.I)
_TR_RE = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.I)
_CELL_RE = re.compile(r"<t[dh]\b([^>]*)>([\s\S]*?)</t[dh]>", re.I)

_DROP_KEYS = (
    "参考文献",
    "参考资料",
    "致谢",
    "作者简介",
    "作者介绍",
    "基金项目",
    "收稿日期",
    "引用格式",
    "中图分类号",
    "references",
    "bibliography",
    "acknowledg",
)

# 只剥标题开头的序号，避免误伤“1998年洪水”这类标题。
_SECTION_PREFIXES = (
    re.compile(r"^第\s*[0-9０-９一二三四五六七八九十百]+\s*[章节部分篇]\s*"),
    re.compile(rf"^[0-9０-９]+(?:[\.．][0-9０-９]+)+(?:\s+|(?=[{_CJK}]))"),
    re.compile(r"^[（(]\s*[0-9０-９一二三四五六七八九十]+\s*[）)]\s*"),
    re.compile(r"^[一二三四五六七八九十百]+[、\.．]\s*"),
    re.compile(r"^[0-9０-９]+[、\.．]\s*"),
    re.compile(rf"^[0-9０-９]+\s+(?=[{_CJK}])"),
    # 正文清洗可能把“1 概述”收成“1概述”；只剥一到两位序号，保留“1998年”。
    re.compile(rf"^[0-9０-９]{{1,2}}(?=[{_CJK}])"),
    # 英文标题如 "1 Introduction"、"2.1 Data"，不剥 "1998 Floods"。
    re.compile(r"^[0-9０-９]{1,2}(?:[\.．][0-9０-９]{1,2})+\s+(?=[A-Za-z])"),
    re.compile(r"^[0-9０-９]{1,2}[\.．]\s*(?=[A-Za-z])"),
    re.compile(r"^[0-9０-９]{1,2}\s+(?=[A-Za-z])"),
)


def compact_cjk_spaces(text: str) -> str:
    return re.sub(rf"(?<=[{_CJK}]){_HS}+(?=[{_CJK}])", "", text)


def _strip_number_cjk_spaces(text: str) -> str:
    text = text.replace("\\~", "~")
    text = re.sub(rf"(?<=[{_CJK}]){_HS}+(?=[{_NUM}])", "", text)
    text = re.sub(rf"(?<=[{_NUM}]){_HS}+(?=[{_CJK}])", "", text)
    return text


def map_outside_math(text: str, fn) -> str:
    parts: list[str] = []
    last = 0
    for match in _MATH_RE.finditer(text):
        parts.append(fn(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(fn(text[last:]))
    return "".join(parts)


def _strip_body_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.split("\n"):
        if _HEADING_RE.match(line):
            lines.append(line)
        else:
            lines.append(_strip_number_cjk_spaces(line))
    return "\n".join(lines)


def normalize_number_cjk_spaces(text: str) -> str:
    """去掉公式和标题以外、汉字与阿拉伯数字之间的空白。"""
    return map_outside_math(text, _strip_body_lines)


def restore_caption_space(text: str) -> str:
    """表题、图题保留“表1 标题”这种编号后的空格，正文不恢复。"""

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)} "

    return _CAPTION_SPACE_RE.sub(repl, text)


def join_number_cjk_linebreaks(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return text
    merged = [lines[0]]
    for current in lines[1:]:
        previous = merged[-1]
        if (
            previous
            and current
            and "|" not in previous
            and "|" not in current
            and "$" not in previous
            and "$" not in current
            and not _HEADING_RE.match(previous)
            and not _HEADING_RE.match(current)
            and (
                (
                    re.search(rf"[{_NUM}]$", previous)
                    and re.match(rf"[{_CJK}]", current)
                )
                or (
                    re.search(rf"[{_CJK}]$", previous)
                    and re.match(rf"[{_NUM}]", current)
                )
            )
        ):
            merged[-1] = previous + current
        else:
            merged.append(current)
    return "\n".join(merged)


def clean_section_title(title: str) -> str:
    original = title.strip()
    title = _TAG_RE.sub("", original)
    title = html.unescape(title).replace("\\", "").replace("*", "")
    title = title.replace("_", "")
    title = compact_cjk_spaces(title)
    title = re.sub(r"\s+", " ", title).strip()
    for _ in range(4):
        updated = title
        for pattern in _SECTION_PREFIXES:
            updated = pattern.sub("", updated).strip()
        if updated == title:
            break
        title = updated
    title = title.strip(" ：:.-")
    return title or re.sub(r"\s+", " ", _TAG_RE.sub("", original)).strip()


def is_dropped_section(path: list[str]) -> bool:
    for part in path:
        folded = compact_cjk_spaces(part).replace(" ", "").lower()
        if any(key in folded for key in _DROP_KEYS):
            return True
    return False


def polish_title(title: str) -> str:
    title = _TAG_RE.sub("", title)
    title = html.unescape(title).replace("\\", "").replace("*", "")
    title = compact_cjk_spaces(title)
    title = normalize_number_cjk_spaces(title)
    title = re.sub(r"\s+", " ", title).strip().strip("#").strip()
    return title


_RUNNING_ID_RE = re.compile(
    r"文章编号\s*\d+\s*-\s*\d+\s*[（(]\s*\d+\s*[）)]\s*\d+\s*-\s*\d+\s*-\s*\d+"
)


def strip_running_article_ids(text: str) -> str:
    """去掉插进正文的期刊页眉，并把被它拆开的词接回去。"""
    text = re.sub(
        rf"([{_CJK}])(?:\s*{_RUNNING_ID_RE.pattern}\s*)+([{_CJK}])",
        r"\1\2",
        text,
    )
    return _RUNNING_ID_RE.sub("", text)


def _cell_text(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    raw = _TAG_RE.sub("", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = compact_cjk_spaces(raw)
    return raw.replace("|", "\\|")


def _attr_int(attrs: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*[\"']?(\d+)", attrs, re.I)
    if not match:
        return 1
    return max(1, int(match.group(1)))


def _table_to_markdown(table_html: str) -> str:
    """按 rowspan / colspan 把单元格放到正确的列，再转成管道表。"""
    row_htmls = _TR_RE.findall(table_html)
    if not row_htmls:
        return table_html
    grid: list[list[str | None]] = []

    def ensure(row: int, col: int) -> None:
        while len(grid) <= row:
            grid.append([])
        while len(grid[row]) <= col:
            grid[row].append(None)

    for row_index, row_html in enumerate(row_htmls):
        column = 0
        for attrs, inner in _CELL_RE.findall(row_html):
            while True:
                ensure(row_index, column)
                if grid[row_index][column] is None:
                    break
                column += 1
            text = _cell_text(inner)
            rowspan = _attr_int(attrs, "rowspan")
            colspan = _attr_int(attrs, "colspan")
            for row in range(row_index, row_index + rowspan):
                for col in range(column, column + colspan):
                    ensure(row, col)
                    if row == row_index:
                        grid[row][col] = text
                    elif grid[row][col] is None:
                        grid[row][col] = ""
            column += colspan
    rows = [[cell or "" for cell in row] for row in grid if any(cell for cell in row)]
    if not rows:
        return table_html
    width = max(len(row) for row in rows)

    def render(row: list[str]) -> str:
        padded = row + [""] * (width - len(row))
        return "| " + " | ".join(padded) + " |"

    lines = [
        render(rows[0]),
        "| " + " | ".join(["---"] * width) + " |",
        *(render(row) for row in rows[1:]),
    ]
    return "\n".join(lines)


def _replace_sup(match: re.Match[str]) -> str:
    inner = re.sub(r"\s+", "", _TAG_RE.sub("", match.group(1)))
    if inner in {"．", "."}:
        return "."
    if re.search(r"[［\[\]］]", inner):
        return ""
    if re.fullmatch(r"[0-9，,、]+", inner) and re.search(r"[，,、]", inner):
        return ""
    if re.fullmatch(r"[0-9+\-]{1,3}", inner):
        return inner.translate(_SUPER_DIGITS)
    return inner


def _replace_sub(match: re.Match[str]) -> str:
    inner = re.sub(r"\s+", "", _TAG_RE.sub("", match.group(1)))
    if inner in {"．", "。"}:
        return "."
    if re.fullmatch(r"[0-9+\-]{1,3}", inner):
        return inner.translate(_SUB_DIGITS)
    return inner


def repair_ocr_markup(text: str) -> str:
    """保留单位上标，修复被下标拆开的小数点，去掉文献角标。"""
    text = _SUB_RE.sub(_replace_sub, text)
    text = _SUP_RE.sub(_replace_sup, text)
    return text


def repair_decimals(text: str) -> str:
    def fix(chunk: str) -> str:
        chunk = re.sub(r"(?<=[0-9])．(?=[0-9])", ".", chunk)
        previous = None
        while previous != chunk:
            previous = chunk
            chunk = re.sub(r"(\d\.\d+?)(?=\d\.)", r"\1 ", chunk)
        return chunk

    return map_outside_math(text, fix)


def html_tables_to_markdown(text: str) -> str:
    return _HTML_TABLE_RE.sub(lambda match: _table_to_markdown(match.group(0)), text)


def _is_garbage_english_heading(title: str) -> bool:
    if re.search(rf"[{_CJK}]", title):
        return False
    tokens = re.findall(r"[A-Za-z]+", title)
    if len(tokens) < 2:
        return False
    singles = sum(1 for token in tokens if len(token) == 1)
    return singles / len(tokens) >= 0.35


def _drop_title_headings(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    title = ""
    first_h1: int | None = None
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = polish_title(match.group(2))
            first_h1 = index
            break
    if first_h1 is None:
        return text, title
    drop = {first_h1}
    # 中文论文题目后常紧跟一条英文译题；英文论文自身的一级标题要保留。
    title_is_chinese = bool(re.search(rf"[{_CJK}]", title))
    for index in range(first_h1 + 1, len(lines)):
        match = _HEADING_RE.match(lines[index])
        if not match:
            continue
        if len(match.group(1)) > 1:
            break
        heading = match.group(2)
        heading_title = polish_title(heading)
        same_title = heading_title.casefold() == title.casefold()
        # 译题通常是完整英文题目；单词章节名如 Introduction 不在这里丢掉。
        tokens = re.findall(r"[A-Za-z]+", heading)
        latin = sum(len(token) for token in tokens)
        cjk = len(re.findall(rf"[{_CJK}]", heading))
        drop_translation = title_is_chinese and len(tokens) >= 3 and latin >= 12 and cjk * 3 < latin
        if _is_garbage_english_heading(heading) or drop_translation or same_title:
            drop.add(index)
            continue
        break
    kept = [line for index, line in enumerate(lines) if index not in drop]
    return "\n".join(kept), title


def _remove_images_and_captions(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if _IMAGE_RE.search(lines[index]) and lines[index].strip().startswith("!"):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines) and _FIG_CAPTION_RE.match(lines[index].strip()):
                index += 1
            continue
        kept.append(_IMAGE_RE.sub("", lines[index]))
        index += 1
    return "\n".join(kept)


def prepare_markdown(raw: str) -> tuple[str, str]:
    """清洗版面噪声，保留表格和公式。返回 (正文, 题目)。"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = strip_running_article_ids(text)
    text = repair_ocr_markup(text)
    text = html_tables_to_markdown(text)
    text = _remove_images_and_captions(text)
    text = _CITATION_LATEX_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text, title = _drop_title_headings(text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[-*_]{3,}", stripped):
            continue
        cleaned_lines.append(line.rstrip())
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = join_number_cjk_linebreaks(text)
    text = normalize_number_cjk_spaces(text)
    text = repair_decimals(text)
    text = restore_caption_space(text)
    return text.strip(), title


def strip_reference_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\n+", text)
    kept: list[str] = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        hits = sum(1 for line in lines if re.match(r"^[\[［]\d+[\]］]", line))
        if hits >= 2 and hits / len(lines) >= 0.6:
            continue
        kept.append(paragraph.strip())
    return "\n\n".join(kept).strip()


def _is_furniture_line(stripped: str) -> bool:
    if re.fullmatch(r"第\s*\d+\s*页|[—\-–]\s*\d{1,4}\s*[—\-–]", stripped):
        return True
    match = re.match(rf"^(?:图|圖|Fig\.?|Figure)\s*[{_NUM}]+[.\s:：]*(.*)$", stripped, re.I)
    if not match:
        return False
    rest = match.group(1)
    if "。" in stripped:
        return False
    # 长句是在讨论文中的图，短行和没有句号的英文图题删掉。
    return not (re.search(r"[.!?]", rest) and len(stripped) > 120)


def strip_furniture_lines(text: str) -> str:
    kept = [line for line in text.splitlines() if not _is_furniture_line(line.strip())]
    return "\n".join(kept)


def _drop_short_figure_lines(text: str) -> str:
    return strip_furniture_lines(text)


def cleanup_record_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _IMAGE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = text.splitlines()
    if lines and _HEADING_RE.match(lines[0].strip()):
        lines = lines[1:]
    text = "\n".join(lines)
    text = _drop_short_figure_lines(text)
    text = strip_reference_paragraphs(text)
    text = join_number_cjk_linebreaks(text)
    text = normalize_number_cjk_spaces(text)
    text = repair_decimals(text)
    text = restore_caption_space(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """按标题切成 (章节路径, 含标题的 Markdown 块)。文首无标题的部分路径为空。"""
    lines = text.splitlines()
    marks: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            marks.append((index, len(match.group(1)), match.group(2)))
    if not marks:
        body = text.strip()
        return [([], body)] if body else []

    sections: list[tuple[list[str], str]] = []
    if marks[0][0] > 0:
        preamble = "\n".join(lines[: marks[0][0]]).strip()
        if preamble:
            sections.append(([], preamble))

    stack: list[tuple[int, str]] = []
    for index, (line_no, level, raw_title) in enumerate(marks):
        title = clean_section_title(raw_title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = marks[index + 1][0] if index + 1 < len(marks) else len(lines)
        block = "\n".join(lines[line_no:end]).strip()
        path = [item for _, item in stack if item]
        if block:
            sections.append((path, block))
    return sections


def split_paragraph_groups(text: str, limit: int) -> list[str]:
    paragraphs = [part for part in re.split(r"\n\n+", text) if part.strip()]
    if not paragraphs:
        return []
    groups: list[str] = []
    buffer: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if len(paragraph) > limit and not buffer:
            groups.append(paragraph.strip())
            continue
        if buffer and size + len(paragraph) > limit:
            groups.append("\n\n".join(buffer).strip())
            buffer = []
            size = 0
        buffer.append(paragraph.strip())
        size += len(paragraph) + 2
    if buffer:
        groups.append("\n\n".join(buffer).strip())
    return groups


_META_LINE_RE = re.compile(
    r"^(?:DOI|doi)\b"
    r"|^(?:关键词|关键字|中图分类号|文献标识码|文章编号|收稿日期|基金项目)\s*[:：]"
    r"|^(?:keywords?|key\s*words)\s*[:：]"
    r"|^(?:received|accepted|revised|published|correspondence|corresponding author)\b",
    re.I,
)
def _strip_abstract_label(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?i)摘\s*要|abstract", stripped):
            continue
        line = re.sub(r"(?i)^(?:摘\s*要|abstract)\s*[:：]\s*", "", line)
        lines.append(line)
    return "\n".join(lines)


def is_mostly_english(text: str) -> bool:
    prose = _MATH_RE.sub(" ", text)
    cjk = len(re.findall(rf"[{_CJK}]", prose))
    latin = len(re.findall(r"[A-Za-z]", prose))
    return latin >= 20 and cjk * 3 < latin


def document_is_english(text: str) -> bool:
    prose = _MATH_RE.sub(" ", text)
    cjk = len(re.findall(rf"[{_CJK}]", prose))
    latin = len(re.findall(r"[A-Za-z]", prose))
    return latin >= 80 and latin > cjk * 2


def paragraph_needs_repair(text: str) -> bool:
    """只有缺句读的正文才送给模型，公式和表格保持原样。"""
    stripped = text.strip()
    if not stripped or stripped.startswith("$$") or stripped.startswith("|"):
        return False
    if stripped.count("|") >= 4:
        return False
    prose = _MATH_RE.sub(" ", stripped)
    gaps = len(re.findall(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])", prose))
    cjk = len(re.findall(rf"[{_CJK}]", prose))
    if cjk < 16:
        return False
    if gaps >= 2:
        return True
    punctuation = prose.count("。") + prose.count("，") + prose.count("；")
    return cjk >= 60 and punctuation * 40 < cjk


def clean_preamble(text: str, keep_english: bool = False) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _META_LINE_RE.match(stripped):
            continue
        if re.fullmatch(r"[（(].+[）)]", stripped) and "。" not in stripped:
            continue
        kept.append(line)
    text = _strip_abstract_label("\n".join(kept))
    paragraphs = [part.strip() for part in re.split(r"\n\n+", text) if part.strip()]
    kept_paragraphs = []
    for paragraph in paragraphs:
        if not keep_english and is_mostly_english(paragraph):
            continue
        if keep_english:
            if re.search(r"@|https?://|orcid", paragraph, re.I):
                continue
            if not re.search(r"[.?!](?:\s|$)", paragraph):
                continue
            words = re.findall(r"[A-Za-z]{2,}", paragraph)
            if len(words) < 8:
                continue
        elif len(paragraph) < 40 and "。" not in paragraph and "，" not in paragraph:
            continue
        kept_paragraphs.append(paragraph)
    return "\n\n".join(kept_paragraphs).strip()


def section_paragraphs(text: str) -> list[tuple[bool, str]]:
    pieces: list[tuple[bool, str]] = []
    for part in re.split(r"\n\n+", text):
        paragraph = part.strip()
        if not paragraph:
            continue
        pieces.append((paragraph_needs_repair(paragraph), paragraph))
    return pieces


def self_check() -> None:
    assert normalize_number_cjk_spaces("1998 年洪水") == "1998年洪水"
    assert normalize_number_cjk_spaces("前 186 年") == "前186年"
    assert normalize_number_cjk_spaces("投入 1.2 亿元") == "投入1.2亿元"
    math_text = "1998 年 $$V = \\frac{F}{C}$$ 共 2 项"
    assert normalize_number_cjk_spaces(math_text) == "1998年 $$V = \\frac{F}{C}$$ 共2项"
    inline = "第 3 节 $x _ { 1 }$ 共 2 项"
    assert normalize_number_cjk_spaces(inline) == "第3节 $x _ { 1 }$ 共2项"
    assert clean_section_title("1.1 纺织印染行业的环境概况") == "纺织印染行业的环境概况"
    assert clean_section_title("一、引言") == "引言"
    assert clean_section_title("（2）筒子染色废水") == "筒子染色废水"
    assert clean_section_title("1998年洪水成因及特点") == "1998年洪水成因及特点"
    assert repair_ocr_markup("面积1000km<sup>2</sup>、水量1亿m<sup>3</sup>") == "面积1000km²、水量1亿m³"
    assert repair_decimals(repair_ocr_markup("0<sub>．</sub>315600<sub>．</sub>57328")) == "0.31560 0.57328"
    assert repair_decimals(repair_ocr_markup("气温18<sub>．</sub>3℃")) == "气温18.3℃"
    assert clean_section_title("1概述") == "概述"
    assert clean_section_title("3 结束语") == "结束语"
    assert clean_section_title("0 引 言") == "引言"
    assert clean_section_title("1 Introduction") == "Introduction"
    assert clean_section_title("2.1 Data and methods") == "Data and methods"
    assert clean_section_title("1998 Floods in China") == "1998 Floods in China"
    assert document_is_english("Flood risk has increased. " * 20)
    assert not document_is_english("洪水风险持续上升。" * 20 + " Flood risk.")
    assert paragraph_needs_repair("近30年来 地方政府投入了较多人力 并开展论证")
    assert not paragraph_needs_repair("近30年来，地方政府投入了较多人力，并开展论证。")
    assert not paragraph_needs_repair("$$\nI_s = K N\n$$")
    assert "摘要正文" in clean_preamble("罗勇钢\n\n摘 要: 摘要正文，用于测试。\n\n关键词: 浊度")
    assert clean_section_title("2.2.1水平潜流人工湿地") == "水平潜流人工湿地"
    table = (
        "<table><tr><td>废水来源</td><td>产生量( $m^3/d$ )</td></tr>"
        "<tr><td>筒染</td><td>733.4</td></tr></table>"
    )
    rendered = html_tables_to_markdown(table)
    assert rendered.splitlines()[0] == "| 废水来源 | 产生量( $m^3/d$ ) |"
    assert "| 筒染 | 733.4 |" in rendered
    grouped = html_tables_to_markdown(
        "<table><tr><td rowspan=\"2\">计算范 围</td><td rowspan=\"2\">汇水面积</td>"
        "<td rowspan=\"2\">地表径流系数</td><td colspan=\"2\">矿坑涌水量</td></tr>"
        "<tr><td>雨季日最大 涌水量</td><td>平均涌 水</td></tr>"
        "<tr><td>境界内</td><td>320830.93</td><td>0.8</td><td>9240</td><td>84.7</td></tr></table>"
    )
    grouped_lines = [line for line in grouped.splitlines() if line.startswith("|") and "---" not in line]
    assert "雨季日最大涌水量" in grouped_lines[1]
    assert grouped_lines[1].rstrip().endswith("平均涌水 |")
    assert grouped_lines[2].split("|")[4].strip() == "9240"
    assert "计算范围" in grouped_lines[0]
    joined = prepare_markdown(
        "通过构造裂文章编号1674-6708（2016）160-0107-01\n\n隙继续入渗，成为矿区地下水的主要补给来源。"
    )[0]
    assert "构造裂隙继续入渗" in joined
    assert "文章编号" not in joined
    caption = restore_caption_space("表1洪水风险\n1998年发生洪水")
    assert caption == "表1 洪水风险\n1998年发生洪水"
    assert is_dropped_section(["讨论", "参 考 文 献"])
    prepared, title = prepare_markdown(
        "# 测试 标题 2020 年\n\n正文 1998 年完成。\n\n![](a.jpg)\n图1 示意\n"
    )
    assert title == "测试标题2020年"
    assert "1998年" in prepared
    assert "![](" not in prepared
    assert "图1" not in prepared
    bilingual, bilingual_title = prepare_markdown(
        "# 洪水风险研究\n\n摘要：本文研究沿海城市的洪水风险，资料覆盖三十年降雨和潮位。\n\n"
        "# Flood Risk Assessment\n\nAbstract: This paper studies flood risk over thirty years of rainfall.\n\n"
        "## 1 引言\n\n沿海城市的洪水风险持续上升，本文利用三十年降雨和潮位资料分析其变化。\n"
    )
    assert bilingual_title == "洪水风险研究"
    assert "Flood Risk" not in bilingual
    assert "引言" in bilingual
    captioned = cleanup_record_text(
        "由图3可知，1981～2010年桂阳烟区烤烟生长期温差呈现缓慢减小的趋势，年平均值为7.46℃。\n"
        "Fig. 3 Variation trend of average temperature difference during 1981-2010.\n"
        "图3 桂阳1981～2010年烤烟大田生长期的平均温差\n"
        "— 12 —\n"
    )
    assert "温差呈现缓慢减小" in captioned
    assert "Fig." not in captioned
    assert "平均温差" not in captioned
    assert "— 12 —" not in captioned
    kept_sentence = cleanup_record_text(
        "Fig. 1 shows the hydrograph recorded at the downstream station during the 1998 flood, "
        "and the peak discharge arrived about six hours after the catchment rainfall ended."
    )
    assert kept_sentence.startswith("Fig. 1 shows")
