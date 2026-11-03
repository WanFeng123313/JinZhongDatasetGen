#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描输入目录下全部 PDF，生成主索引并均分成 N 份。

用法：
  python build_pdf_index.py \\
    --input  /NAS_TYSK/02Data/大模型三水论文/20261102提供 \\
    --output /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1102 \\
    --shards 3

输出（自动创建 --output 目录）：
  all_pdfs.tsv          全量索引
  shard_00.tsv ...      N 个分片（--shards 1 时仅写 all_pdfs.tsv）
  shards_manifest.tsv   分片清单

索引列（TAB）：rel_path  bytes  mtime_ns  batch  lang  doc_id
索引一台建好即可复用；各机 MinerU：
  python mineru_local_multi_gpu.py \\
    --input <本机论文根目录> --file-list <共享>/shard_00.tsv --output ...
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator

INDEX_FIELDS = ("rel_path", "bytes", "mtime_ns", "batch", "lang", "doc_id")


def normalize_rel(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def infer_batch_lang(rel: str) -> tuple[str, str]:
    parts = normalize_rel(rel).split("/")
    for i, p in enumerate(parts):
        if p in ("中文", "英文") and i + 1 < len(parts):
            return parts[i + 1], ("zh" if p == "中文" else "en")
        if p in ("Zw1.1", "HbA1.1", "OpA1.1"):
            return p, ("zh" if p.startswith("Zw") else "en")
    return "unknown", "unknown"


def row_from_pdf(pdf: Path, root: Path) -> dict[str, str]:
    rel = normalize_rel(str(pdf.relative_to(root)))
    batch, lang = infer_batch_lang(rel)
    try:
        st = pdf.stat()
        size = str(st.st_size)
        mtime_ns = str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        size, mtime_ns = "0", "0"
    return {
        "rel_path": rel,
        "bytes": size,
        "mtime_ns": mtime_ns,
        "batch": batch,
        "lang": lang,
        "doc_id": pdf.stem,
    }


def write_index(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in INDEX_FIELDS})
            n += 1
    return n


def iter_index_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        if first.split("\t", 1)[0].strip() == "rel_path":
            f.seek(0)
            for row in csv.DictReader(f, delimiter="\t"):
                rel = normalize_rel(row.get("rel_path", ""))
                if not rel or rel.startswith("#"):
                    continue
                yield {
                    "rel_path": rel,
                    "bytes": row.get("bytes", "0") or "0",
                    "mtime_ns": row.get("mtime_ns", "0") or "0",
                    "batch": row.get("batch", "") or "",
                    "lang": row.get("lang", "") or "",
                    "doc_id": row.get("doc_id", "") or Path(rel).stem,
                }
            return

        def emit(line: str) -> dict[str, str] | None:
            line = line.strip()
            if not line or line.startswith("#"):
                return None
            rel = normalize_rel(line.split("\t")[0])
            if not rel.lower().endswith(".pdf"):
                return None
            batch, lang = infer_batch_lang(rel)
            return {
                "rel_path": rel,
                "bytes": "0",
                "mtime_ns": "0",
                "batch": batch,
                "lang": lang,
                "doc_id": Path(rel).stem,
            }

        row0 = emit(first)
        if row0:
            yield row0
        for line in f:
            row = emit(line)
            if row:
                yield row


def load_pdf_paths(
    file_list: Path,
    root: Path,
    *,
    check_exists: bool = False,
    progress_every: int = 100_000,
    progress_fn=None,
) -> tuple[list[Path], list[str]]:
    """供 mineru_local_multi_gpu 等调用：返回 (PDF 路径, 非法/缺失 rel_path)。

    默认不做逐文件 is_file（NAS 上对百万级路径极慢），只做安全路径拼接。
    需要强校验时传 check_exists=True。
    """
    root = root.resolve()
    out: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()
    n = 0
    for row in iter_index_rows(file_list):
        rel = row["rel_path"]
        if rel in seen:
            continue
        seen.add(rel)
        parts = Path(normalize_rel(rel)).parts
        if not parts or any(p == ".." for p in parts):
            missing.append(rel)
            continue
        pdf = root.joinpath(*parts)
        if check_exists and not pdf.is_file():
            missing.append(rel)
            continue
        out.append(pdf)
        n += 1
        if progress_every and progress_fn and n % progress_every == 0:
            progress_fn(n)
    return out, missing


def build_and_split(input_dir: Path, output_dir: Path, shards: int) -> None:
    root = input_dir.resolve()
    if not root.is_dir():
        sys.exit(f"输入目录不存在: {root}")
    if shards < 1:
        sys.exit("--shards 必须 >= 1")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = output_dir / "all_pdfs.tsv"

    print(f"[1/2] 扫描 PDF: {root}", flush=True)
    rows: list[dict[str, str]] = []
    n = 0
    with master.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for pdf in root.rglob("*"):
            if not (pdf.is_file() and pdf.suffix.lower() == ".pdf"):
                continue
            row = row_from_pdf(pdf, root)
            w.writerow(row)
            rows.append(row)
            n += 1
            if n % 200000 == 0:
                print(f"  ...已扫描 {n}", flush=True)
    print(f"  全量索引: {master}  ({n} 篇)", flush=True)

    if shards == 1:
        print("[2/2] shards=1，仅生成 all_pdfs.tsv", flush=True)
        return

    print(f"[2/2] 均分成 {shards} 份 -> {output_dir}", flush=True)
    width = max(2, len(str(shards - 1)))
    base, rem = divmod(n, shards)
    start = 0
    manifest = output_dir / "shards_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as mf:
        mf.write("shard_file\trows\n")
        for i in range(shards):
            size = base + (1 if i < rem else 0)
            chunk = rows[start:start + size]
            start += size
            path = output_dir / f"shard_{i:0{width}d}.tsv"
            write_index(path, chunk)
            mf.write(f"{path.name}\t{len(chunk)}\n")
            print(f"  {path.name}: {len(chunk)}", flush=True)
    print(f"完成。清单: {manifest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="扫描 PDF 并生成索引/分片（参数尽量简单）",
    )
    ap.add_argument("--input", "-i", type=Path, required=True, help="输入文件夹（递归找 PDF）")
    ap.add_argument("--output", "-o", type=Path, required=True, help="输出文件夹（自动创建）")
    ap.add_argument("--shards", "-n", type=int, required=True, help="待分数量（机器数）；1=只写全量索引")
    args = ap.parse_args()
    build_and_split(args.input, args.output, args.shards)


if __name__ == "__main__":
    main()
