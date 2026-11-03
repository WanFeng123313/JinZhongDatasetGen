#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinerU 3.4.5: recursive PDF -> Markdown, independent local CLI per GPU.
No Router / no --api-url. Each GPU consumes disjoint batches.

支持两种输入：
  1) --input 目录递归扫 PDF（旧行为）
  2) --input <数据集根> + --file-list <索引/分片.tsv>（多机按索引分配）

索引工具见 build_pdf_index.py。
nohup python -u 01_mineru_local_multi_gpu.py > /mnt/sdc/mineru_3gpu_10000.log 2>&1 &
tail -f /mnt/sdc/mineru_3gpu_10000.log
"""
__version__ = '1.2.0'

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from build_pdf_index import load_pdf_paths, normalize_rel

DEFAULT_INPUT = Path('/mnt/sdc/金钟大模型文献/英文10000样品/英文10000样品/pdf')
DEFAULT_OUTPUT = Path('/mnt/sdc/jinzhong_MD_10000_eg')

# MinerU 日志里常见的进度行，例如:
# Submitting batch 1/1 | 1 document ... | task#12 [xxx]
TASK_RE = re.compile(r'task#(\d+)', re.I)
SUBMIT_RE = re.compile(r'Submitting batch', re.I)
ERROR_HINT_RE = re.compile(
    r'(ValueError|CUDA out of memory|Free memory on device|Error:|Traceback)',
    re.I,
)


def has_completed(dst: Path) -> bool:
    if not (dst / '.success').is_file():
        return False
    return any(f.is_file() and f.stat().st_size > 0 for f in dst.rglob('*.md'))


def destination(src: Path, root: Path, output: Path) -> Path:
    relative = src.relative_to(root)
    return output / relative.parent / src.stem


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m{s:02d}s'
    if m:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return f"[{'?' * width}] {done}/?"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({100 * ratio:5.1f}%)"


def count_md_stems(results_dir: Path) -> int:
    """临时 results 下已产出 md 的文档目录数（batch 内实时进度）。"""
    if not results_dir.is_dir():
        return 0
    n = 0
    try:
        for child in results_dir.iterdir():
            if child.is_dir() and any(
                f.is_file() and f.stat().st_size > 0 for f in child.rglob('*.md')
            ):
                n += 1
    except OSError:
        return n
    return n


def run_batch(gpu, number, files, src_root, output, args, log_dir, live, live_lock):
    # Temp is unique per GPU/batch, never shared by two mineru commands.
    with tempfile.TemporaryDirectory(prefix=f'_mineru_gpu{gpu}_', dir=output) as temp:
        tmp = Path(temp)
        staged = tmp / 'input'
        results = tmp / 'results'
        staged.mkdir()
        results.mkdir()
        for pdf in files:
            (staged / pdf.name).symlink_to(pdf.resolve())
        cmd = ['mineru', '-p', str(staged), '-o', str(results),
               '-b', args.backend,
               '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        # prevent inheriting host-configured remote API address
        env.pop('MINERU_API_URL', None)
        log_file = log_dir / f'gpu{gpu}_batch{number:05d}.log'
        start = time.monotonic()
        with live_lock:
            live[gpu] = {
                'batch': number,
                'batch_total': len(files),
                'task_seen': 0,
                'md_ready': 0,
                'results_dir': str(results),
                'last_line': 'starting',
                'started': start,
            }

        print(
            f'[GPU {gpu}] START batch={number:05d} files={len(files)} '
            f'log={log_file}',
            flush=True,
        )

        return_code = -1
        try:
            with log_file.open('w', encoding='utf-8') as log:
                log.write(f'GPU={gpu} count={len(files)} command={cmd}\n')
                log.flush()
                proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    stripped = line.rstrip()
                    task_m = TASK_RE.search(stripped)
                    interesting = bool(
                        task_m or SUBMIT_RE.search(stripped) or ERROR_HINT_RE.search(stripped)
                    )
                    with live_lock:
                        state = live.get(gpu, {})
                        if task_m:
                            state['task_seen'] = max(
                                int(state.get('task_seen', 0)), int(task_m.group(1)),
                            )
                        state['last_line'] = stripped[-160:]
                        state['md_ready'] = count_md_stems(results)
                        live[gpu] = state
                    if interesting:
                        seen = live.get(gpu, {}).get('task_seen', 0)
                        md_n = live.get(gpu, {}).get('md_ready', 0)
                        print(
                            f'[GPU {gpu}] batch={number:05d} '
                            f'{progress_bar(max(seen, md_n), len(files), width=20)} '
                            f'| {stripped[-100:]}',
                            flush=True,
                        )
                return_code = proc.wait()
        except Exception as exc:
            with log_file.open('a', encoding='utf-8') as log:
                log.write(f'EXCEPTION: {exc!r}\n')
            return_code = -1

        success = 0
        failures = []
        for pdf in files:
            source_dir = results / pdf.stem
            # MinerU may create additional subdirs beneath the stem directory.
            good = source_dir.is_dir() and any(
                f.is_file() and f.stat().st_size > 0
                for f in source_dir.rglob('*.md')
            )
            if not good:
                failures.append(pdf)
                continue
            dest = destination(pdf, src_root, output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Existing noncompleted output is left intact on any failure.
                if has_completed(dest):
                    success += 1
                    continue
                if dest.exists():
                    # Keep partial old attempts separately, don't delete them.
                    backup = dest.with_name(dest.name + '.incomplete_' + str(int(time.time_ns())))
                    dest.rename(backup)
                shutil.move(str(source_dir), str(dest))
                (dest / '.success').write_text(json.dumps({
                    'pdf': str(pdf.relative_to(src_root)),
                    'gpu': gpu,
                    'finished_at': datetime.now().isoformat(timespec='seconds'),
                }, ensure_ascii=False), encoding='utf-8')
                success += 1
            except Exception as exc:
                failures.append(pdf)
                with log_file.open('a', encoding='utf-8') as log:
                    log.write(f'OUTPUT ERROR {pdf}: {exc!r}\n')

        with live_lock:
            live.pop(gpu, None)
        return success, failures, return_code, time.monotonic() - start, log_file


def collect_pdfs(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    """返回 (src_root, pdf_list)。优先 --file-list 索引分片。"""
    src_root = args.input.resolve()
    if not src_root.is_dir():
        raise SystemExit(f'Input directory not found: {src_root}')

    if args.file_list:
        file_list = args.file_list.resolve()
        if not file_list.is_file():
            raise SystemExit(f'file-list not found: {file_list}')
        all_pdfs, missing = load_pdf_paths(file_list, src_root)
        if missing:
            print(f'[warn] missing {len(missing)} paths from index (show up to 5):', flush=True)
            for m in missing[:5]:
                print(f'  - {m}', flush=True)
        # 索引模式下以相对路径判重（允许不同目录同名文件名）
        seen_rel: dict[str, Path] = {}
        for pdf in all_pdfs:
            rel = normalize_rel(str(pdf.relative_to(src_root)))
            if rel in seen_rel:
                raise SystemExit(f'Duplicate rel_path in file-list: {rel}')
            seen_rel[rel] = pdf
        print(f'[info] file-list={file_list} loaded={len(all_pdfs)} missing={len(missing)}', flush=True)
    else:
        all_pdfs = sorted(
            p for p in src_root.rglob('*')
            if p.is_file() and p.suffix.lower() == '.pdf'
        )
        # 目录扫描模式保持旧行为：同名文件名冲突则退出
        seen_name: dict[str, Path] = {}
        for pdf in all_pdfs:
            key = pdf.name.casefold()
            if key in seen_name:
                raise SystemExit(
                    f'Duplicate filename across dirs: {seen_name[key]} and {pdf}'
                )
            seen_name[key] = pdf

    if args.limit:
        all_pdfs = all_pdfs[:args.limit]
    return src_root, all_pdfs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                    help='PDF 根目录；与索引配合时为 rel_path 的解析根（数据集根）')
    ap.add_argument('--file-list', type=Path, default=None,
                    help='索引/分片 TSV（或每行一个 rel_path）；指定后只处理列表内 PDF')
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--gpus', default='0,1,2', help='GPU IDs, e.g. 0,1 or 0,1,2')
    ap.add_argument('--batch-size', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--backend', default='hybrid-engine')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    ap.add_argument('--progress-interval', type=float, default=30.0,
                    help='总体进度心跳间隔（秒）；<=0 关闭')
    args = ap.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        ap.error('--batch-size >= 1 and --limit >= 0 required')
    if not (0 < args.gpu_memory_utilization <= 1):
        ap.error('--gpu-memory-utilization must be in (0,1]')
    try:
        gpus = [int(s.strip()) for s in args.gpus.split(',')]
        assert gpus and len(gpus) == len(set(gpus)) and all(x >= 0 for x in gpus)
    except (AssertionError, ValueError):
        ap.error('--gpus should be unique nonnegative IDs separated by commas')

    src_root, all_pdfs = collect_pdfs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / '_logs'
    log_dir.mkdir(exist_ok=True)

    pending = [p for p in all_pdfs if not has_completed(destination(p, src_root, output))]
    skipped = len(all_pdfs) - len(pending)
    total = len(all_pdfs)
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size if pending else 0
    print(
        f'Found: {total}, skipped: {skipped}, pending: {len(pending)}; '
        f'GPUs: {gpus}; batch-size: {args.batch_size}; batches: {n_batches}',
        flush=True,
    )
    print('IMPORTANT: no --api-url, each GPU runs its own local MinerU engine', flush=True)
    print(f'[progress] {progress_bar(skipped, total)} (already done)', flush=True)
    if not pending:
        return
    batches = queue.Queue()
    for index in range(0, len(pending), args.batch_size):
        batches.put((index // args.batch_size, pending[index:index + args.batch_size]))
    lock = threading.Lock()
    live_lock = threading.Lock()
    live: dict[int, dict] = {}
    stats = {'ok': 0, 'failed': 0, 'batches': 0}
    started = time.monotonic()
    stop_heartbeat = threading.Event()
    failed_path = log_dir / 'failed_local.txt'

    def heartbeat():
        while not stop_heartbeat.wait(args.progress_interval):
            with lock:
                ok = stats['ok']
                failed = stats['failed']
                finished_batches = stats['batches']
            in_flight = 0
            parts = []
            with live_lock:
                for gpu, st in sorted(live.items()):
                    md_n = count_md_stems(Path(st.get('results_dir', '')))
                    task_n = int(st.get('task_seen', 0))
                    cur = max(md_n, task_n)
                    in_flight += cur
                    elapsed_b = time.monotonic() - float(st.get('started', started))
                    parts.append(
                        f'GPU{gpu} b{st.get("batch", "?")}:'
                        f'{progress_bar(cur, int(st.get("batch_total", 0)), width=12)}'
                        f' {fmt_duration(elapsed_b)}'
                    )
            done_like = skipped + ok
            # in-flight 只作展示，不计入最终 done，避免重复
            elapsed = max(1, time.monotonic() - started)
            rate = ok / elapsed * 3600
            remain = max(0, len(pending) - ok - failed)
            eta = (remain / (ok / elapsed)) if ok > 0 else float('inf')
            eta_s = fmt_duration(eta) if eta != float('inf') else 'estimating...'
            print(
                f'[progress] {progress_bar(done_like, total)} '
                f'ok+skip={done_like} failed={failed} '
                f'batches={finished_batches}/{n_batches} '
                f'rate={rate:.1f}/h ETA={eta_s}',
                flush=True,
            )
            if parts:
                print('[live] ' + ' | '.join(parts), flush=True)

    # Do not destroy older failure logs, append per run.
    def worker(gpu):
        while True:
            try:
                index, files = batches.get_nowait()
            except queue.Empty:
                return
            try:
                ok, failed, rc, seconds, logfile = run_batch(
                    gpu, index, files, src_root, output, args, log_dir, live, live_lock)
            except Exception as exc:
                ok, failed, rc, seconds, logfile = 0, files, -1, 0.0, 'internal error'
                with (log_dir / 'scheduler_errors.txt').open('a', encoding='utf-8') as err:
                    err.write(f'GPU {gpu} batch {index}: {exc!r}\n')
            with lock:
                stats['ok'] += ok
                stats['failed'] += len(failed)
                stats['batches'] += 1
                if failed:
                    with failed_path.open('a', encoding='utf-8') as out:
                        for p in failed:
                            out.write(f'{datetime.now().isoformat()}\tGPU{gpu}\t{p}\n')
                elapsed = max(1, time.monotonic() - started)
                done_like = skipped + stats['ok']
                print(
                    f'[GPU {gpu}] DONE  batch={index:05d} '
                    f'{progress_bar(ok, len(files), width=20)} exit={rc} '
                    f'time={fmt_duration(seconds)} overall={stats["ok"] / elapsed * 3600:.1f}/h '
                    f'total {progress_bar(done_like, total, width=20)} '
                    f'logs={logfile}',
                    flush=True,
                )
            batches.task_done()

    hb = None
    if args.progress_interval and args.progress_interval > 0:
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    stop_heartbeat.set()
    elapsed = max(1, time.monotonic() - started)
    print(
        f'FINISHED {progress_bar(skipped + stats["ok"], total)} '
        f'ok={stats["ok"]}, skipped={skipped}, failed={stats["failed"]}, '
        f'wall={fmt_duration(elapsed)}, throughput={stats["ok"] / elapsed * 3600:.1f} PDFs/hour',
        flush=True,
    )


if __name__ == '__main__':
    main()
