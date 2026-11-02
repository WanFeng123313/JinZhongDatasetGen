from __future__ import annotations

import base64
from pathlib import Path

SYSTEM_PROMPT = """你是学术论文预训练语料整理程序。只输出一个 JSON 对象，不要输出解释、Markdown 围栏或思考过程。
不得编造论文中不存在的事实，不得把正文缩写成摘要，不得更换实词。

输出格式：
{"records":[{"paper_title":"","text":"","metadata":{"domain":"","subdomain":"","language":"","section_path":["章节"]}}]}

整理规则：
1. 按标题层级切分。一个标题及其直属正文是一条记录；该标题下还有下级标题时，导语单独成条，下级标题再各自成条。没有正文的纯标题不要成条。
2. text 保持原文表述。删除期刊名、期号、页眉页脚、图片、图题、参考文献、作者、单位、作者简介、基金项目、收稿日期。摘要、引言、正文、结论、讨论都要保留。
3. 表格必须保留为 Markdown 管道表，表题（如“表1 …”）留在表格上方，和表格放在同一条记录里。公式必须保留为 LaTeX：行内 $...$，独立公式 $$...$$。不要把表格或公式改写成说明文字，也不要把它们放进代码块。可以收紧公式里明显多余的空格，但不要改符号。只有一两句的正文也要保留。
4. 正文里汉字和阿拉伯数字之间不要留空格。例如“1998 年”“1.2 亿元”“第 3 节”写成“1998年”“1.2亿元”“第3节”。不要删除英文单词之间的空格，不要改公式内部。
5. 去掉残留 HTML 标签。被错误换行拆开的句子要合并。源文经常用空格代替逗号和句号，必须把这种断句空格恢复为逗号或句号，段落之间用一个空行分隔。例如原文“近30年来 地方政府投入人力。结果表明 周期为10a”应写成“近30年来，地方政府投入人力。结果表明，周期为10a”。小数点用英文句点，单位上标保留为 km²、m³。不要改数值、术语、公式和表格。
6. text 不要重复章节标题，不要包含图片链接。标题行里如果粘连了下一级标题，拆成多条记录。
7. 同一篇论文的 paper_title、domain、subdomain、language 必须完全一致。中文论文的 domain 和 subdomain 用简短中文专业词，例如“水文水资源/洪水预报”。英文论文的 domain 和 subdomain 必须用简短英文专业词，例如“Hydrology and Water Resources/Flood Forecasting”。不要使用“工程科技Ⅰ辑”“基础科学”这类专辑名，也不要照搬示例。
8. section_path 是从一级章节到当前章节的标题数组，不包含论文题目。去掉标题前的序号（如“1.1”“一、”“（2）”），保留标题中有意义的年份和名称。
9. language 只能是“中文”或“English”，并与 domain、subdomain 的语言一致。英文论文必须填 English，不要填“英文”。"""


def _language_token(english: bool | None) -> str:
    if english is True:
        return "English"
    if english is False:
        return "中文"
    return ""


def _label_rule(english: bool | None) -> str:
    if english is True:
        return (
            "这篇是英文论文。language 必须填 English，不要填“英文”。"
            "domain 和 subdomain 必须是简短英文专业词，例如 Hydrology and Water Resources、Flood Forecasting。"
            "不要用中文，不要用专辑名，不要照搬示例。"
        )
    if english is False:
        return (
            "这篇是中文论文。language 必须填“中文”。"
            "domain 和 subdomain 用简短中文专业词，例如水文水资源、洪水预报。"
            "不要用“工程科技Ⅰ辑”这类专辑名，不要照搬示例。"
        )
    return (
        "中文论文的 domain、subdomain 用简短中文专业词，language 填“中文”。"
        "英文论文的 domain、subdomain 必须用简短英文专业词，language 填 English，不要填“英文”。"
        "不要用知网专辑名。"
    )


def build_fast_messages(
    head: str,
    source: str,
    title_hint: str,
    paragraphs: list[tuple[int, str]],
    english: bool | None = None,
) -> list[dict[str, str]]:
    hint = title_hint or "（未识别，请从正文判断）"
    system = (
        "你是学术论文校对员。只输出一个 JSON 对象，不要解释。"
        "不要编造事实，不要缩写，不要更换实词、数字、公式和表格。"
    )
    if paragraphs:
        numbered = "\n\n".join(f"[{index}] {text}" for index, text in paragraphs)
        user = (
            "根据论文开头给出分类，并校对下面编号段落。"
            "输出 {\"paper_title\":\"\",\"domain\":\"\",\"subdomain\":\"\",\"language\":\""
            f"{_language_token(english)}\","
            "\"paragraphs\":[{\"id\":0,\"text\":\"\"}]}。\n"
            "paragraphs 必须与编号一一对应，不能增删。汉字之间用于断句的空格改成逗号或句号，"
            "明显缺失的顿号要补上。汉字和数字之间不要留空格。公式、表格和数值保持原样。\n"
            f"{_label_rule(english)}\n"
            f"来源路径：{source}\n候选论文题目：{hint}\n\n论文开头：\n{head}\n\n待校对段落：\n{numbered}"
        )
    else:
        user = (
            "阅读论文开头，只输出 JSON："
            "{\"paper_title\":\"\",\"domain\":\"\",\"subdomain\":\"\",\"language\":\""
            f"{_language_token(english)}\"}}。\n"
            f"{_label_rule(english)}\n"
            f"来源路径：{source}\n候选论文题目：{hint}\n\n论文开头：\n{head}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_single_pass_messages(markdown: str, source: str, title_hint: str) -> list[dict[str, str]]:
    hint = title_hint or "（未识别，请从正文判断）"
    user = (
        f"来源路径只用于判断领域，不要写入 text：{source}\n"
        f"候选论文题目：{hint}。若它准确，所有记录都使用它。\n"
        "汉字之间用于断句的空格改成逗号或句号，不要保留这种空格。\n\n"
        f"论文 Markdown：\n{markdown}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_classify_messages(
    head: str,
    source: str,
    title_hint: str,
    english: bool | None = None,
) -> list[dict[str, str]]:
    hint = title_hint or "（未识别，请从正文判断）"
    user = (
        "阅读论文开头，只输出 JSON 对象："
        '{"paper_title":"","domain":"","subdomain":"","language":"'
        f'{_language_token(english)}"}}。\n'
        f"{_label_rule(english)}\n"
        f"来源路径：{source}\n"
        f"候选论文题目：{hint}\n\n"
        f"论文开头：\n{head}"
    )
    return [
        {"role": "system", "content": "你只输出 JSON 对象，不要解释。"},
        {"role": "user", "content": user},
    ]


def build_chunk_messages(
    fragments: list[tuple[list[str], str]],
    source: str,
    meta: dict[str, str],
) -> list[dict[str, str]]:
    blocks: list[str] = []
    for index, (path, markdown) in enumerate(fragments, start=1):
        hint = " / ".join(path) if path else "（无标题，可能是摘要或文首说明）"
        blocks.append(f"--- 片段{index} 章节路径提示：{hint} ---\n{markdown}")
    body = "\n\n".join(blocks)
    user = (
        "只整理下面这些片段，不要生成片段以外的章节。\n"
        "每个片段前的章节路径提示是该片段已有的标题层级。对应记录的 section_path 必须沿用这个层级；片段内部还有更小的标题时再往后追加。标题如果粘连了下级标题，请拆开。\n"
        "提示语本身不要写入 text。汉字之间用于断句的空格改成逗号或句号。\n"
        f"paper_title 必须是：{meta['paper_title']}\n"
        f"domain 必须是：{meta['domain']}\n"
        f"subdomain 必须是：{meta['subdomain']}\n"
        f"language 必须是：{meta['language']}\n"
        f"来源路径：{source}\n\n"
        f"{body}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_page_messages(title: str, image_paths: list[Path], reject_note: str = "") -> list[dict]:
    system = (
        "你是期刊转页排版整理程序。只输出一个 JSON 对象，不要解释、不要思考过程。"
        '格式：{"markdown":""}。'
        "图片是同一份 PDF 的连续页面。中文期刊通常是双栏：先读完左栏，再读右栏，读完本页再读下一页。"
        "只保留题目所指的这一篇。另一篇文章整段删除，包括它的上接、下转和续文。"
        "“（下转第N页）”不是分段。下一页里，先跳过不属于本题的文字，从本题的续文接着读，并把断开的句子接成一句。"
        "“（上接第N页）”上面、和题目无关的文字属于另一篇，全部删除；标记下面才是本题从第N页接过来的正文。"
        "章节写成 Markdown 标题，一级用 ##，下级用 ###，不要用加粗代替标题。"
        "表格保留为 Markdown 管道表，表题放在表上方。跨列的表头要占对列，不要改、不要丢表内数字，数字中间不要加空格。"
        "公式保留为 LaTeX。去掉页眉、页脚、页码、作者、单位、参考文献和题录。不要缩写，不要编造。"
    )
    instruction = (
        f"论文题目：{title or '（见页面标题）'}\n"
        f"下面共 {len(image_paths)} 页。只输出这篇论文的 Markdown。"
    )
    if reject_note:
        instruction += "\n" + reject_note
    content: list[dict] = [{"type": "text", "text": instruction}]
    for path in image_paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def build_stitch_messages(title: str, image_paths: list[Path], fragment: str) -> list[dict]:
    system = (
        "你只输出一个 JSON 对象，不要解释。"
        '格式：{"sentence":""}。'
        "sentence 是题目这篇论文在“下转”处被截断的那一句，按页面原文字把前后接上。"
        "不要带上另一篇文章的字，不要改数字。"
    )
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"论文题目：{title}\n"
                f"残段：{fragment}\n"
                "请看“下转”紧前面的文字和后一页本题续文，输出补全后的这一句。"
            ),
        }
    ]
    for path in image_paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
