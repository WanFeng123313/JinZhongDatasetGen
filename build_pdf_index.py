#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描输入目录下全部 PDF，生成主索引并均分成 N 份。

用法：
  # 扫目录 + 按篇数均分（旧行为）
  python build_pdf_index.py \\
    --input  /NAS_TYSK/02Data/大模型三水论文/20261102提供 \\
    --output /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1102 \\
    --shards 3

  # 已有 all_pdfs.tsv：按文件大小重切分（不必重扫 PDF）
  # --balance bytes 后默认把每片再按体积从小到大排序，避免 MinerU 队首全是巨本
  python build_pdf_index.py \\
    --from-index /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1103/all_pdfs.tsv \\
    --output /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1104 \\
    --shards 5 \\
    --balance bytes \\
    --shard-order small-first

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
import shutil
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


def row_bytes(row: dict[str, str]) -> int:
    try:
        return max(0, int(row.get("bytes") or 0))
    except ValueError:
        return 0


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


def load_index_rows(index_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in iter_index_rows(index_path):
        rel = row["rel_path"]
        if rel in seen:
            continue
        seen.add(rel)
        rows.append(row)
    return rows


def sort_shard_rows(
    rows: list[dict[str, str]],
    shard_order: str,
) -> list[dict[str, str]]:
    """分片写入前的排序。small-first 避免 FIFO 消费时队首全是巨本。"""
    if shard_order == "as-is":
        return list(rows)
    if shard_order == "small-first":
        return sorted(rows, key=lambda r: (row_bytes(r), r.get("rel_path", "")))
    if shard_order == "large-first":
        return sorted(rows, key=lambda r: (-row_bytes(r), r.get("rel_path", "")))
    raise ValueError(f"unknown shard_order: {shard_order}")


def split_rows(
    rows: list[dict[str, str]],
    shards: int,
    balance: str,
) -> list[list[dict[str, str]]]:
    if shards < 1:
        raise ValueError("shards must be >= 1")
    if shards == 1:
        return [list(rows)]

    if balance == "count":
        n = len(rows)
        base, rem = divmod(n, shards)
        out: list[list[dict[str, str]]] = []
        start = 0
        for i in range(shards):
            size = base + (1 if i < rem else 0)
            out.append(rows[start:start + size])
            start += size
        return out

    if balance != "bytes":
        raise ValueError(f"unknown balance mode: {balance}")

    # 大文件优先 + 贪心放进当前累计 bytes 最小的桶，体积更均衡。
    # 注意：分配顺序是大→小，写入前需再用 shard_order 重排，否则队首全是巨本。
    buckets: list[list[dict[str, str]]] = [[] for _ in range(shards)]
    loads = [0] * shards
    ordered = sorted(rows, key=row_bytes, reverse=True)
    for i, row in enumerate(ordered):
        j = loads.index(min(loads))
        buckets[j].append(row)
        loads[j] += row_bytes(row)
        if (i + 1) % 500000 == 0:
            print(f"  ...已分配 {i + 1}/{len(ordered)}", flush=True)
    return buckets


def write_shards(
    rows: list[dict[str, str]],
    output_dir: Path,
    shards: int,
    balance: str,
    *,
    shard_order: str = "small-first",
    master: Path | None = None,
    copy_master_from: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "all_pdfs.tsv"

    if copy_master_from is not None:
        src = copy_master_from.resolve()
        if src != master_path.resolve():
            print(f"[1/2] 复制全量索引: {src} -> {master_path}", flush=True)
            # CIFS/NAS 上 copy2 的 utime 常失败，只用内容复制
            shutil.copyfile(src, master_path)
        else:
            print(f"[1/2] 全量索引已在输出目录: {master_path}", flush=True)
    elif master is not None:
        # 扫描模式：master 已写好
        print(f"[1/2] 全量索引: {master}  ({len(rows)} 篇)", flush=True)
    else:
        print(f"[1/2] 写入全量索引: {master_path}  ({len(rows)} 篇)", flush=True)
        write_index(master_path, rows)

    if shards == 1:
        print("[2/2] shards=1，仅保留 all_pdfs.tsv", flush=True)
        return

    print(
        f"[2/2] 按 {balance} 分成 {shards} 份，片内顺序={shard_order} -> {output_dir}",
        flush=True,
    )
    chunks = split_rows(rows, shards, balance)
    width = max(2, len(str(shards - 1)))
    manifest = output_dir / "shards_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as mf:
        mf.write("shard_file\trows\tbytes\tfirst_bytes\tlast_bytes\n")
        for i, chunk in enumerate(chunks):
            ordered = sort_shard_rows(chunk, shard_order)
            path = output_dir / f"shard_{i:0{width}d}.tsv"
            write_index(path, ordered)
            total_b = sum(row_bytes(r) for r in ordered)
            first_b = row_bytes(ordered[0]) if ordered else 0
            last_b = row_bytes(ordered[-1]) if ordered else 0
            mf.write(f"{path.name}\t{len(ordered)}\t{total_b}\t{first_b}\t{last_b}\n")
            print(
                f"  {path.name}: rows={len(ordered)}  bytes={total_b / 1e12:.3f} TB  "
                f"avg={total_b / max(len(ordered), 1) / 1e6:.2f} MB  "
                f"first={first_b / 1e6:.2f}MB last={last_b / 1e6:.2f}MB",
                flush=True,
            )
    print(f"完成。清单: {manifest}", flush=True)


def scan_pdfs(input_dir: Path, output_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    root = input_dir.resolve()
    if not root.is_dir():
        sys.exit(f"输入目录不存在: {root}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = output_dir / "all_pdfs.tsv"

    print(f"[scan] 扫描 PDF: {root}", flush=True)
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
    return master, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="扫描 PDF 或从已有索引生成分片（支持按 bytes 均衡）",
    )
    ap.add_argument("--input", "-i", type=Path, default=None, help="输入文件夹（递归找 PDF）；与 --from-index 二选一")
    ap.add_argument(
        "--from-index", type=Path, default=None,
        help="已有索引 TSV（如 all_pdfs.tsv）；指定则不扫盘，直接重切分片",
    )
    ap.add_argument("--output", "-o", type=Path, required=True, help="输出文件夹（自动创建）")
    ap.add_argument("--shards", "-n", type=int, required=True, help="待分数量（机器数）；1=只写全量索引")
    ap.add_argument(
        "--balance", choices=("count", "bytes"), default="bytes",
        help="分片策略：bytes=按文件大小贪心均分（默认）；count=按篇数均分",
    )
    ap.add_argument(
        "--shard-order",
        choices=("small-first", "large-first", "as-is"),
        default="small-first",
        help="每个分片写入顺序；bytes 均分时务必 small-first（默认），避免队首全是巨本",
    )
    args = ap.parse_args()

    if args.shards < 1:
        ap.error("--shards 必须 >= 1")
    if bool(args.input) == bool(args.from_index):
        ap.error("请只指定其中一个：--input（扫盘）或 --from-index（已有索引）")

    output_dir = args.output.resolve()

    if args.from_index is not None:
        index_path = args.from_index.resolve()
        if not index_path.is_file():
            sys.exit(f"索引不存在: {index_path}")
        print(f"[load] 读取索引: {index_path}", flush=True)
        rows = load_index_rows(index_path)
        print(f"  载入 {len(rows)} 篇", flush=True)
        write_shards(
            rows,
            output_dir,
            args.shards,
            args.balance,
            shard_order=args.shard_order,
            copy_master_from=index_path,
        )
        return

    assert args.input is not None
    master, rows = scan_pdfs(args.input, output_dir)
    write_shards(
        rows,
        output_dir,
        args.shards,
        args.balance,
        shard_order=args.shard_order,
        master=master,
    )


if __name__ == "__main__":
    main()
