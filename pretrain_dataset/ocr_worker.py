"""用本机 PaddleOCR PP-StructureV3 把 PDF 解析成 Markdown。"""

from __future__ import annotations

import sys
from pathlib import Path


def _page_markdown(result: object) -> str:
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, str) and markdown.strip():
        return markdown
    if isinstance(markdown, dict):
        for key in ("markdown_texts", "markdown", "text"):
            value = markdown.get(key)
            if isinstance(value, str) and value.strip():
                return value
        pieces = [value for value in markdown.values() if isinstance(value, str) and value.strip()]
        if pieces:
            return "\n\n".join(pieces)
    if isinstance(result, dict):
        return _page_markdown(type("Box", (), result))
    return ""


def parse_pdf(pdf_path: Path, output_path: Path) -> None:
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_formula_recognition=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        wired_table_structure_recognition_model_name="SLANet_plus",
        wireless_table_structure_recognition_model_name="SLANet_plus",
        lang="ch",
        device="cpu",
    )
    pages: list[str] = []
    for result in pipeline.predict(input=str(pdf_path)):
        text = _page_markdown(result).strip()
        if text:
            pages.append(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(pages).strip() + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法: python -m pretrain_dataset.ocr_worker PDF 输出.md")
    parse_pdf(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
