"""Markdown 出现转页标记时，把原 PDF 逐页转成图片，交给多模态模型重读。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from pretrain_dataset.config import Config

logger = logging.getLogger("pretrain_dataset")

_CONTINUE_MARKERS = ("上接第", "下转第", "转下页", "接上页", "接下页", "下接第")
_METHOD_DIRS = {"hybrid_auto", "auto", "ocr", "txt", "vlm"}


def has_page_continuation(text: str) -> bool:
    return any(marker in text for marker in _CONTINUE_MARKERS)


def dataset_pdf_path(markdown_path: Path, input_dir: Path, pdf_dir: Path) -> str:
    """样本里的 PDF 相对路径：`--pdf-dir` 最后一级目录 + 该目录下的相对路径。

    优先读 MinerU 写在文档目录里的 `.success`（字段 `pdf` 相对 PDF 根目录）。
    没有标记时，按 Markdown 相对路径去掉解析子目录后还原 PDF 路径。
    """
    relative = _pdf_rel_from_success(markdown_path, input_dir, pdf_dir)
    if not relative:
        relative = _infer_pdf_rel(markdown_path, input_dir)
    return _join_pdf_root(pdf_dir, relative)


def _join_pdf_root(pdf_dir: Path, relative: str) -> str:
    relative = _clean_rel(relative)
    root_name = pdf_dir.name
    if not root_name or root_name in {".", "/"}:
        root_name = pdf_dir.as_posix().rstrip("/").rsplit("/", 1)[-1]
    if not relative:
        return root_name
    if relative == root_name or relative.startswith(root_name + "/"):
        return relative
    return f"{root_name}/{relative}"


def _pdf_rel_from_success(markdown_path: Path, input_dir: Path, pdf_dir: Path) -> str:
    current = markdown_path.parent
    if not _is_relative_to(current, input_dir):
        return ""
    while True:
        marker = current / ".success"
        if marker.is_file():
            relative = _read_success_pdf(marker, pdf_dir)
            if relative:
                return relative
        if current == input_dir or current.parent == current:
            return ""
        parent = current.parent
        if not _is_relative_to(parent, input_dir):
            return ""
        current = parent


def _read_success_pdf(marker: Path, pdf_dir: Path) -> str:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("pdf")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    path = Path(raw.strip())
    if path.is_absolute():
        try:
            path = path.relative_to(pdf_dir)
        except ValueError:
            return ""
    return _clean_rel(path.as_posix())


def _infer_pdf_rel(markdown_path: Path, input_dir: Path) -> str:
    try:
        relative = markdown_path.relative_to(input_dir)
    except ValueError:
        relative = Path(markdown_path.name)
    parts = [part for part in relative.parts if part not in _METHOD_DIRS]
    path = Path(*parts) if parts else Path(markdown_path.name)
    if len(path.parts) >= 2 and path.parent.name == path.stem:
        path = path.parent.with_name(path.parent.name + ".pdf")
    else:
        path = path.with_suffix(".pdf")
    return path.as_posix()


def _clean_rel(value: str) -> str:
    parts: list[str] = []
    for part in value.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def pdf_file_from_dataset_path(dataset_path: str, pdf_dir: Path) -> Path | None:
    """把样本里的 pdf_path 还原成 `--pdf-dir` 下的文件。"""
    relative = _clean_rel(dataset_path)
    root_name = pdf_dir.name
    prefix = f"{root_name}/"
    if relative.startswith(prefix):
        relative = relative[len(prefix) :]
    if not relative or relative.endswith("/"):
        return None
    candidate = pdf_dir.joinpath(*relative.split("/"))
    if candidate.is_file():
        return candidate
    return None


def locate_pdf(markdown_path: Path, input_dir: Path, pdf_dir: Path) -> Path | None:
    if not pdf_dir.is_dir():
        return None
    named = pdf_file_from_dataset_path(
        dataset_pdf_path(markdown_path, input_dir, pdf_dir),
        pdf_dir,
    )
    if named is not None:
        return named
    try:
        relative = markdown_path.relative_to(input_dir)
    except ValueError:
        relative = Path(markdown_path.name)
    parts = [part for part in relative.parts if part != "hybrid_auto"]
    candidate = pdf_dir.joinpath(*parts).with_suffix(".pdf")
    if candidate.is_file():
        return candidate
    matches = [path for path in pdf_dir.rglob(markdown_path.stem + ".pdf") if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def render_pdf_pages(pdf_path: Path, config: Config, dpi: int = 150, max_pages: int = 12) -> list[Path]:
    """用 pdftoppm 把 PDF 前若干页渲成 JPEG。已有且不旧于 PDF 的缓存直接复用。"""
    cache = config.output_dir / "page_cache" / pdf_path.stem
    cached = _cached_pages(cache, pdf_path)
    if cached:
        return cached[:max_pages]
    cache.mkdir(parents=True, exist_ok=True)
    for old in cache.glob("page-*.jpg"):
        old.unlink()
    logger.info("转页稿渲染 PDF 页面：%s", pdf_path.name)
    try:
        completed = subprocess.run(
            [
                "pdftoppm",
                "-jpeg",
                "-r",
                str(dpi),
                "-jpegopt",
                "quality=82",
                "-f",
                "1",
                "-l",
                str(max_pages),
                str(pdf_path),
                str(cache / "page"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("PDF 转图片失败，沿用原 Markdown：%s (%s)", pdf_path.name, exc)
        return []
    pages = _cached_pages(cache, pdf_path)
    if completed.returncode != 0 or not pages:
        detail = (completed.stderr or completed.stdout or "").strip()[-400:]
        logger.warning("PDF 转图片失败，沿用原 Markdown：%s %s", pdf_path.name, detail)
        return []
    if len(pages) >= max_pages:
        logger.info("PDF 超过 %s 页，只送前 %s 页：%s", max_pages, max_pages, pdf_path.name)
    return pages


def _cached_pages(cache: Path, pdf_path: Path) -> list[Path]:
    if not cache.is_dir():
        return []
    pages = sorted(cache.glob("page-*.jpg"), key=_page_index)
    if not pages:
        return []
    if min(path.stat().st_mtime for path in pages) < pdf_path.stat().st_mtime:
        return []
    return pages


def _page_index(path: Path) -> int:
    suffix = path.stem.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def resolve_ocr_python(config: Config) -> Path | None:
    if config.ocr_python:
        path = Path(config.ocr_python)
        return path if path.is_file() else None
    candidate = Path.home() / "miniconda3/envs/ocr_env/bin/python"
    if candidate.is_file():
        return candidate
    return None


def ocr_pdf_markdown(pdf_path: Path, config: Config) -> str | None:
    python = resolve_ocr_python(config)
    if python is None:
        logger.info("未找到本地 OCR 环境，转页稿仍使用原 Markdown：%s", pdf_path.name)
        return None
    cache = config.output_dir / "ocr_cache" / (pdf_path.stem + ".md")
    if cache.is_file() and cache.stat().st_mtime >= pdf_path.stat().st_mtime and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
    logger.info("转页稿重新 OCR：%s", pdf_path)
    try:
        completed = subprocess.run(
            [str(python), "-m", "pretrain_dataset.ocr_worker", str(pdf_path), str(cache)],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("OCR 失败，沿用原 Markdown：%s (%s)", pdf_path.name, exc)
        return None
    if completed.returncode != 0 or not cache.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        logger.warning("OCR 失败，沿用原 Markdown：%s %s", pdf_path.name, detail)
        return None
    text = cache.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("OCR 结果为空，沿用原 Markdown：%s", pdf_path.name)
        return None
    return text


def self_check() -> None:
    import tempfile

    root = Path("/mnt/sdc/金钟大模型文献/中文10000样品/Pdf")
    md_root = Path("/data/md")
    nested = md_root / "信息科技/科技传播/Z2-10105388/hybrid_auto/Z2-10105388.md"
    assert dataset_pdf_path(nested, md_root, root) == "Pdf/信息科技/科技传播/Z2-10105388.pdf"
    dotted = md_root / "论文.2020/hybrid_auto/论文.2020.md"
    assert dataset_pdf_path(dotted, md_root, root) == "Pdf/论文.2020.pdf"
    flat = md_root / "分类/论文.md"
    assert dataset_pdf_path(flat, md_root, root) == "Pdf/分类/论文.pdf"
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        doc = base / "信息科技" / "Z2-1"
        (doc / "hybrid_auto").mkdir(parents=True)
        markdown = doc / "hybrid_auto" / "Z2-1.md"
        markdown.write_text("# 题\n正文。\n", encoding="utf-8")
        (doc / ".success").write_text(
            json.dumps({"pdf": "信息科技/Z2-1.pdf"}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert dataset_pdf_path(markdown, base, root) == "Pdf/信息科技/Z2-1.pdf"
