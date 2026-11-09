#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理 MinerU 输出目录：只保留 Markdown 与 images，跳过日志/缓存目录。

默认保留：
  - 任意 *.md
  - 名为 images 的目录及其全部内容
  - .success（断点续跑标记；可用 --no-keep-success 去掉）

默认跳过（不进入、不删除）：
  - _logs / __pycache__ / .cache / page_cache / ocr_cache / .git
  - 名称以 _cache 结尾的目录
  - 输出根下的 run_*.log（可用 --clean-root-logs 删除）

会删除 hybrid_auto 等处的：*_middle.json / *_model.json / *_origin.pdf /
*_layout.pdf / content_list*.json 等旁路产物。

示例：
  # 先预览（默认 dry-run）
  python clean_md_output.py --input-dir /nvme/大模型三水论文/sanshui_MD_1104

  # 确认后真正删除
  python clean_md_output.py --input-dir /nvme/大模型三水论文/sanshui_MD_1104 --execute
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


SKIP_DIR_NAMES = frozenset({
    "_logs",
    "__pycache__",
    ".cache",
    "page_cache",
    "ocr_cache",
    ".git",
    ".svn",
})
SKIP_DIR_SUFFIXES = ("_cache",)
KEEP_DIR_NAMES = frozenset({"images"})
KEEP_SUFFIXES = frozenset({".md"})


def fmt_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    x = float(max(0, n))
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}" if u != "B" else f"{int(x)}B"
        x /= 1024.0
    return f"{n}B"


def should_skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    lower = name.lower()
    if lower in SKIP_DIR_NAMES:
        return True
    if any(lower.endswith(suf) for suf in SKIP_DIR_SUFFIXES):
        return True
    return False


def is_keep_dir_name(name: str) -> bool:
    return name in KEEP_DIR_NAMES or name.lower() in KEEP_DIR_NAMES


def under_kept_images(rel: Path) -> bool:
    return any(is_keep_dir_name(part) for part in rel.parts)


def is_keep_file(path: Path, *, keep_success: bool) -> bool:
    name = path.name
    if keep_success and name == ".success":
        return True
    if path.suffix.lower() in KEEP_SUFFIXES:
        return True
    return False


def iter_targets(
    root: Path,
    *,
    keep_success: bool,
    clean_root_logs: bool,
):
    """Yield (path, size) of files to delete. Skip log/cache trees; keep md/images."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        try:
            rel = current.relative_to(root)
        except ValueError:
            dirnames[:] = []
            continue

        # 整棵 images 子树保留：不进入、不删
        if under_kept_images(rel):
            dirnames[:] = []
            continue

        # 剪枝：跳过日志/缓存；images 也不进入（内容全留）
        kept: list[str] = []
        for name in dirnames:
            if should_skip_dir(name) or is_keep_dir_name(name):
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in filenames:
            path = current / name
            if is_keep_file(path, keep_success=keep_success):
                continue
            # 根目录 run_*.log：默认跳过；可选清理
            if current == root and name.startswith("run_") and name.endswith(".log"):
                if not clean_root_logs:
                    continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            yield path, size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="清理 MinerU 输出：保留 *.md 与 images/，跳过 _logs 等目录。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="MinerU Markdown 输出根目录",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正删除；不加则只预览（dry-run）",
    )
    parser.add_argument(
        "--no-keep-success",
        action="store_true",
        help="连 .success 一并删除（会破坏 OCR 断点续跑跳过逻辑）",
    )
    parser.add_argument(
        "--clean-root-logs",
        action="store_true",
        help="同时删除输出根下的 run_*.log（默认保留；_logs/ 始终跳过）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="预览时打印的样例路径条数（默认 20）",
    )
    args = parser.parse_args(argv)

    root = args.input_dir
    if not root.is_dir():
        print(f"[error] 目录不存在: {root}", file=sys.stderr)
        return 1

    keep_success = not args.no_keep_success
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] root={root.resolve()}")
    print(
        f"[info] keep: *.md, images/, "
        f"{'.success' if keep_success else '(no .success)'} | "
        f"skip dirs: {', '.join(sorted(SKIP_DIR_NAMES))} / *_cache"
    )

    started = time.perf_counter()
    n = 0
    total = 0
    samples: list[tuple[str, int]] = []
    errors = 0

    for path, size in iter_targets(
        root,
        keep_success=keep_success,
        clean_root_logs=args.clean_root_logs,
    ):
        n += 1
        total += size
        if len(samples) < max(0, args.sample):
            try:
                samples.append((str(path.relative_to(root)), size))
            except ValueError:
                samples.append((str(path), size))
        if args.execute:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors += 1
                if errors <= 10:
                    print(f"[warn] 删除失败: {path} ({exc})", file=sys.stderr)

        if n % 5000 == 0:
            print(f"[progress] scanned={n} size={fmt_bytes(total)} elapsed={time.perf_counter()-started:.1f}s")

    elapsed = time.perf_counter() - started
    print(f"[summary] files={'deleted' if args.execute else 'would_delete'}={n} "
          f"size={fmt_bytes(total)} errors={errors} elapsed={elapsed:.1f}s")
    if samples:
        print("[sample]")
        for rel, size in samples:
            print(f"  {fmt_bytes(size):>10}  {rel}")
    if not args.execute and n:
        print("[hint] 确认无误后加 --execute 执行删除")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
