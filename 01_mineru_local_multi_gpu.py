#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinerU 3.4.5: recursive PDF -> Markdown, independent local CLI per GPU.
No Router / no --api-url. Each GPU consumes disjoint batches.

支持两种输入：
  1) --input 目录递归扫 PDF（旧行为）
  2) --input <数据集根> + --file-list <索引/分片.tsv>（多机按索引分配）

索引工具见 build_pdf_index.py。
"""
__version__ = '1.3.2'

import argparse
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from build_pdf_index import load_pdf_paths, normalize_rel

_RUN_ID_RE = re.compile(r'[^A-Za-z0-9._-]+')


def sanitize_run_id(value: str) -> str:
    cleaned = _RUN_ID_RE.sub('_', (value or '').strip()).strip('._-')
    return cleaned or 'default'


def resolve_run_id(args: argparse.Namespace) -> str:
    """日志子目录名：--run-id > file-list 文件名(stem) > 主机名。"""
    if args.run_id:
        return sanitize_run_id(args.run_id)
    if args.file_list is not None:
        return sanitize_run_id(Path(args.file_list).stem)
    return sanitize_run_id(socket.gethostname())

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


def log(msg: str = '') -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def has_completed(dst: Path) -> bool:
    if not (dst / '.success').is_file():
        return False
    return any(f.is_file() and f.stat().st_size > 0 for f in dst.rglob('*.md'))


def destination(src: Path, root: Path, output: Path) -> Path:
    relative = src.relative_to(root)
    return output / relative.parent / src.stem


def dest_key(src: Path, root: Path) -> str:
    """与 destination() 相对 output 的路径一致，用于快速跳过集合。"""
    relative = src.relative_to(root)
    return normalize_rel(str(relative.parent / src.stem))


def completed_dest_keys(output: Path) -> set[str]:
    """扫一遍输出目录中的 .success，避免对每个 PDF 单独 stat。"""
    keys: set[str] = set()
    if not output.is_dir():
        return keys
    for marker in output.rglob('.success'):
        if not marker.is_file():
            continue
        try:
            keys.add(normalize_rel(str(marker.parent.relative_to(output))))
        except ValueError:
            continue
    return keys


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
    # Temp on local disk (not NAS output): unique per GPU/batch.
    temp_root = Path(args.temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'_mineru_gpu{gpu}_', dir=str(temp_root)) as temp:
        tmp = Path(temp)
        staged = tmp / 'input'
        results = tmp / 'results'
        staged.mkdir()
        results.mkdir()
        missing_now: list[Path] = []
        present: list[Path] = []
        for pdf in files:
            # 索引模式下启动时不做 is_file；缺失文件在此失败并记入 failures
            if pdf.is_file():
                (staged / pdf.name).symlink_to(pdf)
                present.append(pdf)
            else:
                missing_now.append(pdf)
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

        log(
            f'[GPU {gpu}] START batch={number:05d} files={len(files)} '
            f'present={len(present)} missing={len(missing_now)} log={log_file}'
        )

        return_code = 0
        try:
            with log_file.open('w', encoding='utf-8') as lf:
                lf.write(
                    f'GPU={gpu} count={len(files)} present={len(present)} '
                    f'missing={len(missing_now)} command={cmd}\n'
                )
                lf.flush()
                if present:
                    proc = subprocess.Popen(
                        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        lf.write(line)
                        lf.flush()
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
                            log(
                                f'[GPU {gpu}] batch={number:05d} '
                                f'{progress_bar(max(seen, md_n), len(files), width=20)} '
                                f'| {stripped[-100:]}'
                            )
                    return_code = proc.wait()
                else:
                    lf.write('SKIP mineru: all files missing on disk\n')
        except Exception as exc:
            with log_file.open('a', encoding='utf-8') as lf:
                lf.write(f'EXCEPTION: {exc!r}\n')
            return_code = -1

        success = 0
        failures = list(missing_now)
        missing_set = set(missing_now)
        for pdf in files:
            if pdf in missing_set:
                continue
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
                with log_file.open('a', encoding='utf-8') as lf:
                    lf.write(f'OUTPUT ERROR {pdf}: {exc!r}\n')

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
        log(f'[info] loading file-list (no per-file NAS stat): {file_list}')
        t0 = time.monotonic()
        all_pdfs, missing = load_pdf_paths(
            file_list,
            src_root,
            check_exists=False,
            progress_every=100_000,
            progress_fn=lambda n: log(f'  ...index rows {n}'),
        )
        if missing:
            log(f'[warn] invalid {len(missing)} paths from index (show up to 5):')
            for m in missing[:5]:
                log(f'  - {m}')
        # 索引模式下以相对路径判重（允许不同目录同名文件名）
        seen_rel: dict[str, Path] = {}
        for pdf in all_pdfs:
            rel = normalize_rel(str(pdf.relative_to(src_root)))
            if rel in seen_rel:
                raise SystemExit(f'Duplicate rel_path in file-list: {rel}')
            seen_rel[rel] = pdf
        log(
            f'[info] file-list loaded={len(all_pdfs)} invalid={len(missing)} '
            f'in {fmt_duration(time.monotonic() - t0)}'
        )
    else:
        log(f'[info] scanning PDF under {src_root}')
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
    ap.add_argument(
        '--run-id', default=None,
        help='_logs 子目录名；默认用 --file-list 文件名(stem)，否则用主机名；多机共用 output 时互不覆盖',
    )
    ap.add_argument(
        '--temp-dir', type=Path, default=None,
        help='本地临时目录（默认 /tmp/mineru_local_batches）；MinerU 中间结果写这里，勿指向 NAS',
    )
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

    log(f'[info] mineru_local_multi_gpu.py v{__version__}')
    if args.temp_dir is None:
        args.temp_dir = Path(tempfile.gettempdir()) / 'mineru_local_batches'
    args.temp_dir = args.temp_dir.resolve()
    try:
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        probe = args.temp_dir / f'.write_probe_{os.getpid()}'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise SystemExit(f'temp-dir not writable: {args.temp_dir} ({exc})') from exc
    log(f'[info] temp-dir={args.temp_dir}')

    run_id = resolve_run_id(args)
    src_root, all_pdfs = collect_pdfs(args)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        # 多机共用 output 时：_logs/<run_id>/ 隔离，避免互相覆盖
        log_dir = output / '_logs' / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f'output not writable (NAS 权限/IO?): {output} ({exc}); '
            f'可先写到本地盘再同步'
        ) from exc
    log(f'[info] run-id={run_id} log-dir={log_dir}')

    log('[info] scanning output for .success (one pass)')
    t0 = time.monotonic()
    done_keys = completed_dest_keys(output)
    if done_keys:
        pending = [p for p in all_pdfs if dest_key(p, src_root) not in done_keys]
    else:
        pending = list(all_pdfs)
    skipped = len(all_pdfs) - len(pending)
    log(
        f'[info] completed markers={len(done_keys)} skipped={skipped} '
        f'in {fmt_duration(time.monotonic() - t0)}'
    )
    total = len(all_pdfs)
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size if pending else 0
    log(
        f'Found: {total}, skipped: {skipped}, pending: {len(pending)}; '
        f'GPUs: {gpus}; batch-size: {args.batch_size}; batches: {n_batches}'
    )
    log('IMPORTANT: no --api-url, each GPU runs its own local MinerU engine')
    log(f'[progress] {progress_bar(skipped, total)} (already done)')
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
    stop_workers = threading.Event()
    failed_path = log_dir / 'failed_local.txt'

    def heartbeat():
        while not stop_heartbeat.wait(args.progress_interval):
            with lock:
                ok = stats['ok']
                failed = stats['failed']
                finished_batches = stats['batches']
            parts = []
            with live_lock:
                for gpu, st in sorted(live.items()):
                    md_n = count_md_stems(Path(st.get('results_dir', '')))
                    task_n = int(st.get('task_seen', 0))
                    cur = max(md_n, task_n)
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
            log(
                f'[progress] {progress_bar(done_like, total)} '
                f'ok+skip={done_like} failed={failed} '
                f'batches={finished_batches}/{n_batches} '
                f'rate={rate:.1f}/h ETA={eta_s}'
            )
            if parts:
                log('[live] ' + ' | '.join(parts))

    # Do not destroy older failure logs, append per run.
    def worker(gpu):
        while not stop_workers.is_set():
            try:
                index, files = batches.get_nowait()
            except queue.Empty:
                return
            try:
                ok, failed, rc, seconds, logfile = run_batch(
                    gpu, index, files, src_root, output, args, log_dir, live, live_lock)
            except Exception as exc:
                ok, failed, rc, seconds = 0, files, -1, 0.0
                logfile = f'ERROR {exc!r}'
                log(f'[GPU {gpu}] ERROR batch={index:05d} {exc!r}')
                try:
                    with (log_dir / 'scheduler_errors.txt').open('a', encoding='utf-8') as err:
                        err.write(f'GPU {gpu} batch {index}: {exc!r}\n')
                except OSError as log_exc:
                    log(f'[warn] cannot write scheduler_errors: {log_exc!r}')
                # NAS EIO / ENOSPC：停止继续烧完所有 batch
                if isinstance(exc, OSError) and getattr(exc, 'errno', None) in (5, 28):
                    if not stop_workers.is_set():
                        stop_workers.set()
                        log(f'[fatal] filesystem error ({exc!r}); stop remaining batches')
            with lock:
                stats['ok'] += ok
                stats['failed'] += len(failed)
                stats['batches'] += 1
                if failed:
                    try:
                        with failed_path.open('a', encoding='utf-8') as out:
                            for p in failed:
                                out.write(f'{datetime.now().isoformat()}\tGPU{gpu}\t{p}\n')
                    except OSError as log_exc:
                        log(f'[warn] cannot write failed_local: {log_exc!r}')
                elapsed = max(1, time.monotonic() - started)
                done_like = skipped + stats['ok']
                log(
                    f'[GPU {gpu}] DONE  batch={index:05d} '
                    f'{progress_bar(ok, len(files), width=20)} exit={rc} '
                    f'time={fmt_duration(seconds)} overall={stats["ok"] / elapsed * 3600:.1f}/h '
                    f'total {progress_bar(done_like, total, width=20)} '
                    f'logs={logfile}'
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
    log(
        f'FINISHED {progress_bar(skipped + stats["ok"], total)} '
        f'ok={stats["ok"]}, skipped={skipped}, failed={stats["failed"]}, '
        f'wall={fmt_duration(elapsed)}, throughput={stats["ok"] / elapsed * 3600:.1f} PDFs/hour'
    )


if __name__ == '__main__':
    main()
