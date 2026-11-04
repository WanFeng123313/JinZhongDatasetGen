#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinerU 3.4.5: recursive PDF -> Markdown, independent local CLI per GPU.
No Router / no --api-url. Each GPU consumes disjoint batches.

支持两种输入：
  1) --input 目录递归扫 PDF（旧行为）
  2) --input <数据集根> + --file-list <索引/分片.tsv>（多机按索引分配）

海量论文（含 pdf_index_1103 大文件在前的分片）建议：
  --batch-size 200 --mineru-chunk-size 25 --max-bytes 10485760 --order small-first
  脚本会把 ≤10MiB 先做、更大的延后；成功后在 file-list TSV 写入
  ocr_status=ok / ocr_finished_at=...（同 shard 只应一台机器写）。

索引工具见 build_pdf_index.py。
"""
__version__ = '1.5.0'

import argparse
import csv
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from build_pdf_index import normalize_rel, row_bytes

# 写回 --file-list TSV 的进度列（首次 flush 时自动追加）
OCR_STATUS_COL = 'ocr_status'
OCR_FINISHED_COL = 'ocr_finished_at'
OCR_OK = 'ok'

_RUN_ID_RE = re.compile(r'[^A-Za-z0-9._-]+')


def sanitize_run_id(value: str) -> str:
    cleaned = _RUN_ID_RE.sub('_', (value or '').strip()).strip('._-')
    return cleaned or 'default'


def copy_tree_data(src: Path, dst: Path) -> None:
    """Copy directory tree with file contents only (no chmod/utime/xattr).

    CIFS mounts with ``nounix`` often reject ``chmod`` (EPERM). ``shutil.move`` /
    ``copy2`` / default ``copytree`` then fail even after data was written.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(str(dst))
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst if rel == Path('.') else dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(Path(root) / name, target_dir / name)


def resolve_run_id(args: argparse.Namespace) -> str:
    """日志子目录名：--run-id > file-list 文件名(stem) > 主机名。"""
    if args.run_id:
        return sanitize_run_id(args.run_id)
    if args.file_list is not None:
        return sanitize_run_id(Path(args.file_list).stem)
    return sanitize_run_id(socket.gethostname())

DEFAULT_INPUT = Path('/mnt/sdc/金钟大模型文献/英文10000样品/英文10000样品/pdf')
DEFAULT_OUTPUT = Path('/mnt/sdc/jinzhong_MD_10000_eg22')

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


def kill_process_group(proc: subprocess.Popen | None, grace_sec: float = 8.0) -> None:
    """mineru 会拉起临时 fast_api / VLLM::EngineCore；必须按进程组回收，避免孤儿占显存。"""
    if proc is None or proc.pid is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + (grace_sec if sig == signal.SIGTERM else 2.0)
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)


def warn_orphan_mineru_services() -> None:
    """提示可能残留的 mineru-api（不自动杀，以免误伤手动服务）。"""
    try:
        out = subprocess.check_output(
            ['pgrep', '-af', r'mineru\.cli\.fast_api|mineru-api'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return
    log(f'[warn] found {len(lines)} leftover mineru-api process(es); '
        f'they can occupy GPU. If unintended, run: pkill -f mineru.cli.fast_api')
    for ln in lines[:5]:
        log(f'  {ln[:160]}')


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


def _stage_name(index: int, pdf: Path) -> str:
    """Flat staging name unique within a batch (avoid basename collisions)."""
    return f'{index:04d}_{pdf.name}'


def _run_mineru_chunk(
    *,
    gpu: int,
    batch_number: int,
    chunk_index: int,
    chunk_files: list[tuple[int, Path]],
    results: Path,
    temp_root: Path,
    args: argparse.Namespace,
    log_file: Path,
    live: dict,
    live_lock: threading.Lock,
    batch_total: int,
    batch_started: float,
) -> int:
    """Run one mineru CLI on a small chunk; return process exit code."""
    with tempfile.TemporaryDirectory(
        prefix=f'_chunk_g{gpu}_b{batch_number}_c{chunk_index}_',
        dir=str(temp_root),
    ) as ctemp:
        staged = Path(ctemp) / 'input'
        staged.mkdir()
        for idx, pdf in chunk_files:
            link = staged / _stage_name(idx, pdf)
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(pdf)

        cmd = [
            'mineru', '-p', str(staged), '-o', str(results),
            '-b', args.backend,
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env.pop('MINERU_API_URL', None)

        return_code = 0
        proc: subprocess.Popen | None = None
        try:
            with log_file.open('a', encoding='utf-8') as lf:
                lf.write(
                    f'\n--- chunk={chunk_index} files={len(chunk_files)} '
                    f'command={cmd}\n'
                )
                lf.flush()
                proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True,
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
                        # 进度：已落盘 md 数（跨 chunk 累加在 results 下）
                        md_n = count_md_stems(results)
                        if task_m:
                            # chunk 内 task# + 本 batch 已完成 md，作展示用
                            state['task_seen'] = max(md_n, int(task_m.group(1)))
                        else:
                            state['task_seen'] = max(int(state.get('task_seen', 0)), md_n)
                        state['md_ready'] = md_n
                        state['last_line'] = stripped[-160:]
                        state['started'] = batch_started
                        state['batch'] = batch_number
                        state['batch_total'] = batch_total
                        state['results_dir'] = str(results)
                        live[gpu] = state
                    if interesting:
                        md_n = live.get(gpu, {}).get('md_ready', 0)
                        log(
                            f'[GPU {gpu}] batch={batch_number:05d} '
                            f'chunk={chunk_index} '
                            f'{progress_bar(md_n, batch_total, width=20)} '
                            f'| {stripped[-100:]}'
                        )
                return_code = proc.wait()
        except Exception as exc:
            with log_file.open('a', encoding='utf-8') as lf:
                lf.write(f'CHUNK EXCEPTION chunk={chunk_index}: {exc!r}\n')
            return_code = -1
        finally:
            kill_process_group(proc)
        return return_code


def run_batch(gpu, number, files, src_root, output, args, log_dir, live, live_lock):
    # Temp on local disk (not NAS output): unique per GPU/batch.
    temp_root = Path(args.temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'_mineru_gpu{gpu}_', dir=str(temp_root)) as temp:
        tmp = Path(temp)
        results = tmp / 'results'
        results.mkdir()
        missing_now: list[Path] = []
        present: list[tuple[int, Path]] = []
        for i, pdf in enumerate(files):
            # 索引模式下启动时不做 is_file；缺失文件在此失败并记入 failures
            if pdf.is_file():
                present.append((i, pdf))
            else:
                missing_now.append(pdf)

        log_file = log_dir / f'gpu{gpu}_batch{number:05d}.log'
        start = time.monotonic()
        chunk_size = max(1, int(args.mineru_chunk_size))
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
            f'present={len(present)} missing={len(missing_now)} '
            f'chunk-size={chunk_size} log={log_file}'
        )

        return_code = 0
        with log_file.open('w', encoding='utf-8') as lf:
            lf.write(
                f'GPU={gpu} count={len(files)} present={len(present)} '
                f'missing={len(missing_now)} chunk_size={chunk_size} '
                f'backend={args.backend} util={args.gpu_memory_utilization}\n'
            )

        if present:
            for c_i in range(0, len(present), chunk_size):
                chunk = present[c_i:c_i + chunk_size]
                rc = _run_mineru_chunk(
                    gpu=gpu,
                    batch_number=number,
                    chunk_index=c_i // chunk_size,
                    chunk_files=chunk,
                    results=results,
                    temp_root=temp_root,
                    args=args,
                    log_file=log_file,
                    live=live,
                    live_lock=live_lock,
                    batch_total=len(files),
                    batch_started=start,
                )
                if rc != 0:
                    return_code = rc
                    with log_file.open('a', encoding='utf-8') as lf:
                        lf.write(
                            f'[warn] chunk {c_i // chunk_size} exit={rc}; '
                            f'continue next chunk (isolate blast radius)\n'
                        )
        else:
            with log_file.open('a', encoding='utf-8') as lf:
                lf.write('SKIP mineru: all files missing on disk\n')

        ok_pdfs: list[Path] = []
        failures = list(missing_now)
        missing_set = set(missing_now)
        for i, pdf in enumerate(files):
            if pdf in missing_set:
                continue
            # Prefer staged unique name; fall back to raw stem (compat)
            source_dir = results / Path(_stage_name(i, pdf)).stem
            if not source_dir.is_dir():
                source_dir = results / pdf.stem
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
                if has_completed(dest):
                    ok_pdfs.append(pdf)
                    continue
                if dest.exists():
                    backup = dest.with_name(dest.name + '.incomplete_' + str(int(time.time_ns())))
                    dest.rename(backup)
                # NAS/CIFS: copy contents only; avoid shutil.move/copy2 chmod EPERM
                copy_tree_data(source_dir, dest)
                shutil.rmtree(source_dir, ignore_errors=True)
                (dest / '.success').write_text(json.dumps({
                    'pdf': str(pdf.relative_to(src_root)),
                    'gpu': gpu,
                    'finished_at': datetime.now().isoformat(timespec='seconds'),
                }, ensure_ascii=False), encoding='utf-8')
                ok_pdfs.append(pdf)
            except Exception as exc:
                failures.append(pdf)
                with log_file.open('a', encoding='utf-8') as lf:
                    lf.write(f'OUTPUT ERROR {pdf}: {exc!r}\n')

        with live_lock:
            live.pop(gpu, None)
        return ok_pdfs, failures, return_code, time.monotonic() - start, log_file


class FileListLedger:
    """加载/回写 --file-list TSV；成功篇目打 ocr_status=ok。

    同一 shard 文件只应被一台机器上的一个调度进程写入。
    """

    def __init__(self, path: Path, rows: list[dict[str, str]], fieldnames: list[str]):
        self.path = path
        self.rows = rows
        self.fieldnames = list(fieldnames)
        self._by_rel: dict[str, int] = {}
        for i, row in enumerate(rows):
            rel = normalize_rel(row.get('rel_path', ''))
            if rel:
                self._by_rel[rel] = i
        self._dirty = False
        self._lock = threading.Lock()
        for col in (OCR_STATUS_COL, OCR_FINISHED_COL):
            if col not in self.fieldnames:
                self.fieldnames.append(col)

    @classmethod
    def load(cls, path: Path) -> 'FileListLedger':
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f'file-list not found: {path}')
        rows: list[dict[str, str]] = []
        with path.open('r', encoding='utf-8', newline='') as f:
            first = f.readline()
            if not first:
                return cls(path, [], ['rel_path', 'bytes', 'mtime_ns', 'batch', 'lang', 'doc_id',
                                      OCR_STATUS_COL, OCR_FINISHED_COL])
            if first.split('\t', 1)[0].strip() == 'rel_path':
                f.seek(0)
                reader = csv.DictReader(f, delimiter='\t')
                fieldnames = list(reader.fieldnames or [])
                seen: set[str] = set()
                for raw in reader:
                    rel = normalize_rel(raw.get('rel_path', ''))
                    if not rel or rel.startswith('#') or rel in seen:
                        if rel and rel in seen:
                            raise SystemExit(f'Duplicate rel_path in file-list: {rel}')
                        continue
                    seen.add(rel)
                    row = {k: (raw.get(k) or '') for k in fieldnames}
                    row['rel_path'] = rel
                    if 'bytes' not in row:
                        row['bytes'] = '0'
                    rows.append(row)
                return cls(path, rows, fieldnames)

            # 无表头：每行一个 rel_path（可带其它 TAB 列，忽略）
            fieldnames = ['rel_path', 'bytes', 'mtime_ns', 'batch', 'lang', 'doc_id']
            seen = set()

            def emit(line: str) -> dict[str, str] | None:
                line = line.strip()
                if not line or line.startswith('#'):
                    return None
                rel = normalize_rel(line.split('\t')[0])
                if not rel.lower().endswith('.pdf') or rel in seen:
                    return None
                seen.add(rel)
                return {
                    'rel_path': rel,
                    'bytes': '0',
                    'mtime_ns': '0',
                    'batch': '',
                    'lang': '',
                    'doc_id': Path(rel).stem,
                }

            row0 = emit(first)
            if row0:
                rows.append(row0)
            for line in f:
                row = emit(line)
                if row:
                    rows.append(row)
            return cls(path, rows, fieldnames)

    def is_ok(self, rel: str) -> bool:
        i = self._by_rel.get(normalize_rel(rel))
        if i is None:
            return False
        return (self.rows[i].get(OCR_STATUS_COL) or '').strip().lower() == OCR_OK

    def count_ok(self) -> int:
        return sum(1 for r in self.rows if (r.get(OCR_STATUS_COL) or '').strip().lower() == OCR_OK)

    def mark_ok_many(self, rels: list[str]) -> int:
        if not rels:
            return 0
        when = datetime.now().isoformat(timespec='seconds')
        n = 0
        with self._lock:
            for rel in rels:
                i = self._by_rel.get(normalize_rel(rel))
                if i is None:
                    continue
                row = self.rows[i]
                if (row.get(OCR_STATUS_COL) or '').strip().lower() == OCR_OK:
                    continue
                row[OCR_STATUS_COL] = OCR_OK
                row[OCR_FINISHED_COL] = when
                n += 1
            if n:
                self._dirty = True
        return n

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._write_unlocked()
            self._dirty = False

    def _write_unlocked(self) -> None:
        fields = list(self.fieldnames)
        for col in (OCR_STATUS_COL, OCR_FINISHED_COL):
            if col not in fields:
                fields.append(col)
        self.fieldnames = fields
        tmp = self.path.with_name(f'{self.path.name}.tmp.{os.getpid()}')
        with tmp.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(
                f, fieldnames=fields, delimiter='\t', lineterminator='\n',
                extrasaction='ignore',
            )
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k, '') for k in fields})
        try:
            os.replace(tmp, self.path)
        except OSError:
            # CIFS 上 replace 偶发失败：退化为内容覆盖
            shutil.copyfile(tmp, self.path)
            tmp.unlink(missing_ok=True)


def collect_pdfs(
    args: argparse.Namespace,
) -> tuple[Path, list[tuple[Path, int]], FileListLedger | None]:
    """返回 (src_root, [(pdf, bytes), ...], ledger|None)。优先 --file-list。"""
    src_root = args.input.resolve()
    if not src_root.is_dir():
        raise SystemExit(f'Input directory not found: {src_root}')

    sized: list[tuple[Path, int]] = []
    ledger: FileListLedger | None = None
    if args.file_list:
        file_list = args.file_list.resolve()
        log(f'[info] loading file-list (preserve columns, no per-file NAS stat): {file_list}')
        t0 = time.monotonic()
        ledger = FileListLedger.load(file_list)
        missing: list[str] = []
        already_ok = 0
        for i, row in enumerate(ledger.rows):
            rel = row['rel_path']
            parts = Path(normalize_rel(rel)).parts
            if not parts or any(p == '..' for p in parts):
                missing.append(rel)
                continue
            pdf = src_root.joinpath(*parts)
            sized.append((pdf, row_bytes(row)))
            if ledger.is_ok(rel):
                already_ok += 1
            if (i + 1) % 100_000 == 0:
                log(f'  ...index rows {i + 1}')
        if missing:
            log(f'[warn] invalid {len(missing)} paths from index (show up to 5):')
            for m in missing[:5]:
                log(f'  - {m}')
        log(
            f'[info] file-list loaded={len(sized)} tsv_ok={already_ok} invalid={len(missing)} '
            f'in {fmt_duration(time.monotonic() - t0)}'
        )
    else:
        log(f'[info] scanning PDF under {src_root}')
        all_pdfs = sorted(
            p for p in src_root.rglob('*')
            if p.is_file() and p.suffix.lower() == '.pdf'
        )
        seen_name: dict[str, Path] = {}
        for pdf in all_pdfs:
            key = pdf.name.casefold()
            if key in seen_name:
                raise SystemExit(
                    f'Duplicate filename across dirs: {seen_name[key]} and {pdf}'
                )
            seen_name[key] = pdf
            try:
                sized.append((pdf, max(0, pdf.stat().st_size)))
            except OSError:
                sized.append((pdf, 0))

    if args.limit:
        sized = sized[:args.limit]
    return src_root, sized, ledger


def order_and_defer(
    pending: list[tuple[Path, int]],
    *,
    order: str,
    max_bytes: int,
) -> tuple[list[Path], list[tuple[Path, int]]]:
    """按体积排序；超大文件挪到队尾（仍会处理，只是延后）。

    返回 (ordered_paths, deferred_entries)。
    """
    items = list(pending)
    if order == 'small-first':
        items.sort(key=lambda x: (x[1], str(x[0])))
    elif order == 'large-first':
        items.sort(key=lambda x: (-x[1], str(x[0])))
    # order == 'as-is': keep index/scan order

    if max_bytes <= 0:
        return [p for p, _ in items], []

    primary = [(p, b) for p, b in items if b <= max_bytes]
    deferred = [(p, b) for p, b in items if b > max_bytes]
    # 延后段内部同样按 order（small-first 时先啃「中等偏大」）
    return [p for p, _ in primary] + [p for p, _ in deferred], deferred


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
    ap.add_argument('--gpus', default='1,2', help='GPU IDs, e.g. 0,1 or 0,1,2')
    ap.add_argument(
        '--batch-size', type=int, default=200,
        help='调度队列每批文件数（落盘/失败记账粒度）；真正一次 mineru 调用见 --mineru-chunk-size',
    )
    ap.add_argument(
        '--mineru-chunk-size', type=int, default=25,
        help='单次 mineru CLI 处理篇数；小 chunk 隔离 hybrid EngineCore 挂掉后的连坐',
    )
    ap.add_argument(
        '--max-bytes', type=int, default=10_485_760,
        help='大于该字节的 PDF 延后到队尾（0=关闭）；默认 10MiB，避免巨本拖垮小文件吞吐',
    )
    ap.add_argument(
        '--order', choices=('as-is', 'small-first', 'large-first'), default='small-first',
        help='处理顺序；海量英文论文建议 small-first（先吃小文件抬吞吐）',
    )
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--backend', default='hybrid-engine')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.3)
    ap.add_argument('--progress-interval', type=float, default=30.0,
                    help='总体进度心跳间隔（秒）；<=0 关闭')
    args = ap.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        ap.error('--batch-size >= 1 and --limit >= 0 required')
    if args.mineru_chunk_size < 1:
        ap.error('--mineru-chunk-size >= 1 required')
    if args.max_bytes < 0:
        ap.error('--max-bytes >= 0 required')
    if not (0 < args.gpu_memory_utilization <= 1):
        ap.error('--gpu-memory-utilization must be in (0,1]')
    try:
        gpus = [int(s.strip()) for s in args.gpus.split(',')]
        assert gpus and len(gpus) == len(set(gpus)) and all(x >= 0 for x in gpus)
    except (AssertionError, ValueError):
        ap.error('--gpus should be unique nonnegative IDs separated by commas')

    log(f'[info] mineru_local_multi_gpu.py v{__version__}')
    warn_orphan_mineru_services()
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
    src_root, sized, ledger = collect_pdfs(args)
    all_pdfs = [p for p, _ in sized]
    size_map = {p: b for p, b in sized}
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
    if ledger is not None:
        log(
            f'[info] file-list ledger={ledger.path} '
            f'(will set {OCR_STATUS_COL}={OCR_OK} on success; one writer per shard)'
        )

    log('[info] scanning output for .success (one pass)')
    t0 = time.monotonic()
    done_keys = completed_dest_keys(output)
    pending_sized: list[tuple[Path, int]] = []
    skip_tsv = 0
    skip_success = 0
    backfill_rels: list[str] = []
    for p, b in sized:
        try:
            rel = normalize_rel(str(p.relative_to(src_root)))
        except ValueError:
            rel = ''
        tsv_ok = bool(ledger and rel and ledger.is_ok(rel))
        disk_ok = dest_key(p, src_root) in done_keys
        if tsv_ok:
            skip_tsv += 1
            continue
        if disk_ok:
            skip_success += 1
            if ledger is not None and rel:
                backfill_rels.append(rel)
            continue
        pending_sized.append((p, b))
    if ledger is not None and backfill_rels:
        n_bf = ledger.mark_ok_many(backfill_rels)
        if n_bf:
            ledger.flush()
            log(f'[info] backfilled {n_bf} {OCR_STATUS_COL}=ok from existing .success')
    skipped = skip_tsv + skip_success
    log(
        f'[info] skip tsv_ok={skip_tsv} .success={skip_success} '
        f'completed_markers={len(done_keys)} pending={len(pending_sized)} '
        f'in {fmt_duration(time.monotonic() - t0)}'
    )

    pending, deferred = order_and_defer(
        pending_sized, order=args.order, max_bytes=args.max_bytes,
    )
    if deferred:
        deferred_path = log_dir / 'deferred_large.txt'
        with deferred_path.open('w', encoding='utf-8') as df:
            for p, b in deferred:
                try:
                    rel = normalize_rel(str(p.relative_to(src_root)))
                except ValueError:
                    rel = str(p)
                df.write(f'{b}\t{rel}\n')
        log(
            f'[info] deferred large (>{args.max_bytes} B): {len(deferred)} '
            f'-> end of queue; list={deferred_path}'
        )

    total = len(all_pdfs)
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size if pending else 0
    log(
        f'Found: {total}, skipped: {skipped}, pending: {len(pending)}; '
        f'GPUs: {gpus}; batch-size: {args.batch_size}; '
        f'mineru-chunk-size: {args.mineru_chunk_size}; '
        f'order: {args.order}; max-bytes: {args.max_bytes}; batches: {n_batches}'
    )
    log('IMPORTANT: no --api-url, each GPU runs its own local MinerU engine')
    log(f'[progress] {progress_bar(skipped, total)} (already done)')
    if not pending:
        if ledger is not None:
            ledger.flush()
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
            ok_pdfs: list[Path] = []
            try:
                ok_pdfs, failed, rc, seconds, logfile = run_batch(
                    gpu, index, files, src_root, output, args, log_dir, live, live_lock)
                ok = len(ok_pdfs)
            except Exception as exc:
                ok, failed, rc, seconds = 0, files, -1, 0.0
                ok_pdfs = []
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
            if ledger is not None and ok_pdfs:
                rels = []
                for p in ok_pdfs:
                    try:
                        rels.append(normalize_rel(str(p.relative_to(src_root))))
                    except ValueError:
                        continue
                try:
                    marked = ledger.mark_ok_many(rels)
                    ledger.flush()
                    if marked:
                        log(f'[GPU {gpu}] tsv marked {OCR_STATUS_COL}={OCR_OK} x{marked}')
                except OSError as mark_exc:
                    log(f'[warn] cannot update file-list TSV: {mark_exc!r}')
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
    if ledger is not None:
        try:
            ledger.flush()
        except OSError as exc:
            log(f'[warn] final file-list flush failed: {exc!r}')
    elapsed = max(1, time.monotonic() - started)
    log(
        f'FINISHED {progress_bar(skipped + stats["ok"], total)} '
        f'ok={stats["ok"]}, skipped={skipped}, failed={stats["failed"]}, '
        f'wall={fmt_duration(elapsed)}, throughput={stats["ok"] / elapsed * 3600:.1f} PDFs/hour'
    )
    if ledger is not None:
        log(f'[info] file-list {OCR_STATUS_COL}=ok total={ledger.count_ok()} path={ledger.path}')


if __name__ == '__main__':
    main()
