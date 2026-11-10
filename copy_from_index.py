#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按索引/分片 TSV 复制 PDF，保持原始相对目录结构。

与之前 nvme / server12 拷贝相同做法：
  - 读 TSV 的 rel_path（首列，有表头则跳过）
  - 源：{input}/{rel_path} → 目标：{output}/{rel_path}
  - 多线程并行；已存在且同大小则 skip
  - 先写 .partial 再 os.replace，避免半截文件
  - NAS/CIFS 友好：只用内容复制，不依赖 copy2 的 chmod/utime

用法：
  # 拷若干分片到本机 nvme（目录结构与论文根一致）
  python copy_from_index.py \\
    --input  /NAS_TYSK/02Data/大模型三水论文/20261102提供 \\
    --output /home/dongzhen/nvme/大模型三水论文/20261102提供 \\
    --file-list /path/to/large_ge_10m.tsv \\
    --file-list /path/to/shard_00.tsv \\
    --file-list /path/to/shard_01.tsv \\
    --workers 16 \\
    --log-dir /home/dongzhen/nvme/_copy_logs

  # 也可一次传多个，或传整个索引目录里的通配（shell 展开）
  python copy_from_index.py -i "$SRC" -o "$DEST" \\
    --file-list "$IDX"/shard_0{0,1,2,3}.tsv "$IDX"/large_ge_10m.tsv

  # 顺带把用到的 TSV 拷到目标下的子目录
  python copy_from_index.py ... --copy-index-to pdf_index_main12_large10m

  # 只估体积 / 干跑
  python copy_from_index.py ... --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from build_pdf_index import normalize_rel

CHUNK = 1024 * 1024


def load_rels(paths: list[Path]) -> list[str]:
    """按 file-list 顺序去重加载 rel_path。"""
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"file-list not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
            if not first:
                continue
            # 有表头则跳过；否则当作数据行
            if first.split("\t", 1)[0].strip() != "rel_path":
                rel = normalize_rel(first.split("\t", 1)[0])
                if rel and rel not in seen:
                    seen.add(rel)
                    out.append(rel)
            for line in f:
                rel = normalize_rel(line.split("\t", 1)[0])
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                out.append(rel)
    return out


def estimate_bytes(paths: list[Path]) -> tuple[int, int]:
    """用 TSV 第二列 bytes 估总体积（缺列则跳过）。返回 (n_rows, bytes)。"""
    n = 0
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
            if not first:
                continue
            has_header = first.split("\t", 1)[0].strip() == "rel_path"
            lines = [] if has_header else [first]
            lines.extend(f)
            for line in lines:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0].strip():
                    continue
                n += 1
                if len(parts) > 1:
                    try:
                        total += max(0, int(parts[1]))
                    except ValueError:
                        pass
    return n, total


def copy_one(src_root: Path, dst_root: Path, rel: str, chunk: int = CHUNK) -> tuple[str, int, str, str | None]:
    src = src_root / rel
    dst = dst_root / rel
    try:
        if not src.is_file():
            return "missing", 0, rel, "source not found"
        if dst.is_file():
            try:
                ss, ds = src.stat().st_size, dst.stat().st_size
                if ss == ds and ds > 0:
                    return "skip", 0, rel, None
            except OSError:
                pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".partial")
        with src.open("rb") as fsrc, tmp.open("wb") as fdst:
            while True:
                buf = fsrc.read(chunk)
                if not buf:
                    break
                fdst.write(buf)
        os.replace(tmp, dst)
        return "ok", dst.stat().st_size, rel, None
    except Exception as e:  # noqa: BLE001 — 单文件失败记入 fail，不中断整批
        try:
            p = dst.with_name(dst.name + ".partial")
            if p.exists():
                p.unlink()
        except OSError:
            pass
        return "fail", 0, rel, f"{type(e).__name__}: {e}"


def fmt_bytes(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f}TB"
    if n >= 1e9:
        return f"{n / 1e9:.2f}GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f}MB"
    return f"{n}B"


def run_copy(
    src_root: Path,
    dst_root: Path,
    rels: list[str],
    *,
    workers: int,
    log_dir: Path | None,
    progress_every: int,
    dry_run: bool,
) -> int:
    total = len(rels)
    print(
        f"[info] copy total={total:,} workers={workers} "
        f"src={src_root} dst={dst_root}"
        + (" DRY-RUN" if dry_run else ""),
        flush=True,
    )
    if dry_run:
        for rel in rels[:20]:
            print(f"  would copy: {rel}", flush=True)
        if total > 20:
            print(f"  ... and {total - 20:,} more", flush=True)
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)
    err_path = prog_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        err_path = log_dir / "copy_from_index.err"
        prog_path = log_dir / "copy_from_index.progress"
        # 追加失败明细
        err_f = err_path.open("a", encoding="utf-8")
    else:
        err_f = None

    ok = fail = skip = missing = 0
    bytes_ok = 0
    t0 = time.time()
    lock = threading.Lock()
    done = 0

    def on_result(status: str, size: int, rel: str, err: str | None) -> None:
        nonlocal done, ok, fail, skip, missing, bytes_ok
        with lock:
            done += 1
            if status == "ok":
                ok += 1
                bytes_ok += size
            elif status == "skip":
                skip += 1
            elif status == "missing":
                missing += 1
                if err_f is not None:
                    err_f.write(f"MISSING\t{rel}\t{err}\n")
                    err_f.flush()
            else:
                fail += 1
                msg = f"FAIL\t{rel}\t{err}"
                print(msg, file=sys.stderr, flush=True)
                if err_f is not None:
                    err_f.write(msg + "\n")
                    err_f.flush()
            if done % progress_every == 0 or done == total:
                elapsed = max(1e-6, time.time() - t0)
                rate = done / elapsed
                line = (
                    f'{time.strftime("%Y-%m-%d %H:%M:%S")} '
                    f"done={done}/{total} ok={ok} skip={skip} "
                    f"missing={missing} fail={fail} "
                    f"copied={fmt_bytes(bytes_ok)} "
                    f"rate={rate:.1f} files/s "
                    f"write={bytes_ok / elapsed / 1e6:.1f}MB/s "
                    f"eta={(total - done) / max(rate, 1e-6) / 3600:.2f}h"
                )
                print(line, flush=True)
                if prog_path is not None:
                    prog_path.write_text(line + "\n", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(copy_one, src_root, dst_root, rel) for rel in rels]
            for fut in as_completed(futs):
                on_result(*fut.result())
    finally:
        if err_f is not None:
            err_f.close()

    elapsed = max(1e-6, time.time() - t0)
    print(
        f"[done] ok={ok} skip={skip} missing={missing} fail={fail} "
        f"copied={fmt_bytes(bytes_ok)} in {elapsed / 3600:.2f}h "
        f"({ok / elapsed:.1f} files/s)",
        flush=True,
    )
    if log_dir is not None:
        print(f"[info] progress={prog_path} errors={err_path}", flush=True)
    return 1 if (fail or missing) else 0


def copy_index_files(file_lists: list[Path], dest_subdir: Path) -> None:
    dest_subdir.mkdir(parents=True, exist_ok=True)
    for src in file_lists:
        dst = dest_subdir / src.name
        # 内容复制，避开 CIFS copy2 chmod
        data = src.read_bytes()
        tmp = dst.with_name(dst.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        print(f"[info] copied index {src.name} -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按索引/分片 TSV 复制 PDF，保持原始相对目录结构",
    )
    ap.add_argument(
        "--input", "-i", type=Path, required=True,
        help="论文根目录（索引 rel_path 的解析根）",
    )
    ap.add_argument(
        "--output", "-o", type=Path, required=True,
        help="目标根目录（写出 {output}/{rel_path}）",
    )
    ap.add_argument(
        "--file-list", "-f", type=Path, nargs="+", action="append", required=True,
        help="索引或分片 TSV（可多次指定；也支持一次跟多个路径）",
    )
    ap.add_argument("--workers", "-j", type=int, default=16, help="并行线程数（默认 16）")
    ap.add_argument(
        "--log-dir", type=Path, default=None,
        help="进度/失败日志目录；默认 {output}/_copy_logs",
    )
    ap.add_argument(
        "--progress-every", type=int, default=200,
        help="每处理 N 个文件打一行进度（默认 200）",
    )
    ap.add_argument(
        "--copy-index-to", type=str, default=None,
        help="把用到的 TSV 也拷到 {output}/<该相对子目录>/（如 pdf_index_main12_large10m）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只列前几条与体积估计，不拷贝")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers 必须 >= 1")

    # flatten: action=append + nargs=+ → list[list[Path]]
    file_lists: list[Path] = []
    for group in args.file_list:
        file_lists.extend(group)
    # 去重但保序（同一路径写两次不重复读）
    seen_fl: set[Path] = set()
    uniq_lists: list[Path] = []
    for p in file_lists:
        rp = p.resolve()
        if rp in seen_fl:
            continue
        seen_fl.add(rp)
        uniq_lists.append(rp)

    src_root = args.input.resolve()
    dst_root = args.output.resolve()
    if not src_root.is_dir():
        sys.exit(f"输入目录不存在: {src_root}")

    n_est, b_est = estimate_bytes(uniq_lists)
    print(
        f"[info] file-lists={len(uniq_lists)} tsv_rows≈{n_est:,} "
        f"bytes≈{fmt_bytes(b_est)}",
        flush=True,
    )
    for p in uniq_lists:
        print(f"  - {p}", flush=True)

    rels = load_rels(uniq_lists)
    print(f"[info] unique rel_path={len(rels):,}", flush=True)

    if args.copy_index_to and not args.dry_run:
        copy_index_files(uniq_lists, dst_root / args.copy_index_to)

    log_dir = args.log_dir
    if log_dir is None and not args.dry_run:
        log_dir = dst_root / "_copy_logs"

    rc = run_copy(
        src_root,
        dst_root,
        rels,
        workers=args.workers,
        log_dir=log_dir,
        progress_every=max(1, args.progress_every),
        dry_run=args.dry_run,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
