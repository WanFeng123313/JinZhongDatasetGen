"""Markdown 出现转页标记时，把原 PDF 逐页转成图片，交给多模态模型重读。"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from pretrain_dataset.config import Config

logger = logging.getLogger("pretrain_dataset")

_CONTINUE_MARKERS = ("上接第", "下转第", "转下页", "接上页", "接下页", "下接第")


def has_page_continuation(text: str) -> bool:
    return any(marker in text for marker in _CONTINUE_MARKERS)


def locate_pdf(markdown_path: Path, input_dir: Path, pdf_dir: Path) -> Path | None:
    if not pdf_dir.is_dir():
        return None
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
