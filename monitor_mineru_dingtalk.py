#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定期解析 MinerU run_*.log，把进度推到钉钉自定义机器人。

依赖：仅标准库。Webhook / 加签密钥用环境变量或参数传入，勿写进仓库。

用法：
  export DINGTALK_WEBHOOK='https://oapi.dingtalk.com/robot/send?access_token=xxx'
  export DINGTALK_SECRET='SECxxx'   # 若机器人开启了加签；未开启可省略

  # 前台试跑（每 10 分钟推一次；也可 --once 只推一条）
  python monitor_mineru_dingtalk.py \\
    --log-glob '/NAS_TYSK/02Data/大模型三水论文/run_shard*.log' \\
    --interval 600

  # 后台
  nohup python -u monitor_mineru_dingtalk.py \\
    --log-glob '/NAS_TYSK/02Data/大模型三水论文/run_shard*.log' \\
    --interval 1800 \\
    > /tmp/monitor_mineru_dingtalk.log 2>&1 &
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from glob import glob
from pathlib import Path

PROGRESS_RE = re.compile(
    r'\[progress\].*?ok\+skip=(\d+)\s+failed=(\d+)\s+batches=(\d+)/(\d+)'
    r'(?:\s+rate=([0-9.]+)/h)?(?:\s+ETA=([^\s]+))?'
)
LIVE_RE = re.compile(r'\[live\]\s+(.+)$')
STATUS_RE = re.compile(
    r'\[([^\]]+)\]\s+'
    r'(\[(?:GPU\s+\d+|info)\]\s+(?:START batch=|waiting engine-init|acquired engine-init).+)$'
)
FOUND_RE = re.compile(
    r'Found:\s*(\d+),\s*skipped:\s*(\d+),\s*pending:\s*(\d+);.*?'
    r'order:\s*([^;]+);\s*max-bytes:\s*(\d+)'
)
VERSION_RE = re.compile(r'mineru_local_multi_gpu\.py\s+(v[\d.]+)')
START_RE = re.compile(r'^\[([^\]]+)\].*mineru_local_multi_gpu\.py\s+v')
TS_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]')


def parse_log(path: Path) -> dict:
    """从单个 run log 提取最近进度。"""
    info: dict = {
        'path': str(path),
        'name': path.name,
        'mtime': datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds'),
        'version': '',
        'started': '',
        'found': None,
        'skipped': None,
        'pending': None,
        'order': '',
        'max_bytes': '',
        'ok_skip': None,
        'failed': None,
        'batches': '',
        'rate': '',
        'eta': '',
        'live': '',
        'status': '',
        'last_line': '',
        'last_ts': '',
    }
    try:
        # 大日志：头读启动配置（Found/order/version），尾读最新进度，避免只读尾部丢总数
        size = path.stat().st_size
        head_n = 64_000
        tail_n = 512_000
        with path.open('rb') as f:
            head = f.read(min(size, head_n))
            if size > head_n + tail_n:
                f.seek(-tail_n, os.SEEK_END)
                tail = f.read()
                raw = (head + b'\n' + tail).decode('utf-8', errors='ignore')
            elif size > head_n:
                f.seek(head_n)
                raw = (head + f.read()).decode('utf-8', errors='ignore')
            else:
                raw = head.decode('utf-8', errors='ignore')
    except OSError as exc:
        info['error'] = str(exc)
        return info

    lines = raw.splitlines()
    if not lines:
        info['error'] = 'empty'
        return info
    info['last_line'] = lines[-1][-200:]
    m_ts = TS_RE.match(lines[-1])
    if m_ts:
        info['last_ts'] = m_ts.group(1)

    # Found/version 在启动段；大文件时来自文件头
    for L in lines:
        m = START_RE.search(L)
        if m and not info['started']:
            info['started'] = m.group(1)
        m = VERSION_RE.search(L)
        if m:
            info['version'] = m.group(1)
        m = FOUND_RE.search(L)
        if m:
            info['found'] = int(m.group(1))
            info['skipped'] = int(m.group(2))
            info['pending'] = int(m.group(3))
            info['order'] = m.group(4).strip()
            info['max_bytes'] = m.group(5)

    best_prog = None  # 优先带非 0 rate 的 progress；否则用最新一条
    for L in reversed(lines):
        if not info['live']:
            m = LIVE_RE.search(L)
            if m:
                info['live'] = m.group(1).strip()[:180]
        if not info['status']:
            m = STATUS_RE.search(L)
            if m:
                info['status'] = m.group(2).strip()[:200]
        m = PROGRESS_RE.search(L)
        if m:
            rate = m.group(5) or ''
            cand = {
                'ok_skip': int(m.group(1)),
                'failed': int(m.group(2)),
                'batches': f'{m.group(3)}/{m.group(4)}',
                'rate': rate + ('/h' if rate else ''),
                'eta': m.group(6) or '',
            }
            if best_prog is None:
                best_prog = cand
            if rate and rate != '0.0':
                best_prog = cand
                break
        if info['live'] and best_prog and best_prog.get('rate') not in ('', '0.0/h'):
            break
    if best_prog:
        info.update(best_prog)
    return info


def build_markdown(infos: list[dict], title: str) -> str:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    parts = [f'### {title}', f'- 时间：{now}', '']
    total_ok = 0
    total_failed = 0
    for info in infos:
        if info.get('error'):
            parts.append(f"#### `{info['name']}`")
            parts.append(f"- 读日志失败：{info['error']}")
            parts.append('')
            continue
        ok = info.get('ok_skip') or 0
        failed = info.get('failed') or 0
        total_ok += ok
        total_failed += failed
        found = info.get('found')
        pct = f'{100.0 * ok / found:.2f}%' if found else 'n/a'
        parts.append(f"#### `{info['name']}` {info.get('version') or ''}")
        parts.append(
            f"- 进度：ok+skip=**{ok}** / {found or '?'} ({pct})  "
            f"failed={failed}  batches={info.get('batches') or '?'}"
        )
        parts.append(
            f"- rate={info.get('rate') or '?'}  ETA={info.get('eta') or '?'}  "
            f"order={info.get('order') or '?'}"
        )
        if info.get('live'):
            parts.append(f"- live：{info['live']}")
        elif info.get('status'):
            parts.append(f"- 状态：{info['status']}")
        parts.append(f"- 日志最新：{info.get('last_ts') or info.get('mtime')}")
        parts.append('')
    parts.append(f'**合计** ok+skip={total_ok}  failed={total_failed}')
    return '\n'.join(parts)


def dingtalk_sign(secret: str) -> tuple[str, str]:
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f'{timestamp}\n{secret}'
    digest = hmac.new(
        secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256,
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(digest))
    return timestamp, sign


def send_dingtalk(webhook: str, secret: str | None, title: str, text: str) -> None:
    url = webhook
    if secret:
        ts, sign = dingtalk_sign(secret)
        sep = '&' if '?' in url else '?'
        url = f'{url}{sep}timestamp={ts}&sign={sign}'
    body = {
        'msgtype': 'markdown',
        'markdown': {'title': title, 'text': text},
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json; charset=utf-8'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
    if payload.get('errcode', 0) != 0:
        raise RuntimeError(f'DingTalk API error: {payload}')


def collect_logs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for s in sorted(glob(pat)):
            p = Path(s).resolve()
            key = str(p)
            if key in seen:
                continue
            if p.is_file():
                seen.add(key)
                paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--log-glob', action='append', default=None,
        help='run 日志 glob，可多次；默认看环境变量 MINERU_LOG_GLOB 或常见路径',
    )
    ap.add_argument('--interval', type=int, default=1800, help='推送间隔秒，默认 1800（30 分钟）')
    ap.add_argument('--once', action='store_true', help='只推送一次后退出')
    ap.add_argument('--webhook', default=os.environ.get('DINGTALK_WEBHOOK', ''),
                    help='钉钉机器人 webhook；默认读 DINGTALK_WEBHOOK')
    ap.add_argument('--secret', default=os.environ.get('DINGTALK_SECRET', ''),
                    help='加签密钥；默认读 DINGTALK_SECRET')
    ap.add_argument('--title', default='MinerU 进度', help='钉钉消息标题')
    ap.add_argument('--dry-run', action='store_true', help='只打印，不发送')
    args = ap.parse_args()

    patterns = args.log_glob
    if not patterns:
        env_glob = os.environ.get('MINERU_LOG_GLOB', '').strip()
        if env_glob:
            patterns = [env_glob]
        else:
            patterns = ['/NAS_TYSK/02Data/大模型三水论文/run_shard*.log']

    if not args.webhook and not args.dry_run:
        raise SystemExit('请设置 --webhook 或环境变量 DINGTALK_WEBHOOK（也可用 --dry-run 试跑）')
    if args.interval < 60 and not args.once:
        raise SystemExit('--interval 建议 >= 60 秒，避免触发钉钉限流')

    secret = args.secret or None

    def tick() -> None:
        paths = collect_logs(patterns)
        if not paths:
            text = f'### {args.title}\n\n未找到日志：{patterns}'
        else:
            infos = [parse_log(p) for p in paths]
            text = build_markdown(infos, args.title)
        print(datetime.now().isoformat(timespec='seconds'), 'push', flush=True)
        print(text, flush=True)
        if not args.dry_run:
            send_dingtalk(args.webhook, secret, args.title, text)
            print('dingtalk ok', flush=True)

    if args.once:
        tick()
        return

    print(
        f'monitoring {patterns} every {args.interval}s '
        f'webhook={"set" if args.webhook else "dry"}',
        flush=True,
    )
    while True:
        try:
            tick()
        except Exception as exc:
            print(f'[error] {exc!r}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
