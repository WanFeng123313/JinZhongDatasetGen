#!/usr/bin/env bash
# MinerU 钉钉进度监控：start / status / stop / once
# Webhook / 加签密钥用环境变量或本地 dingtalk.env（勿提交仓库）。
#
#   cp dingtalk.env.example dingtalk.env   # 填入真实值
#   ./start_monitor_dingtalk.sh            # 后台启动
#   ./start_monitor_dingtalk.sh status|stop|once
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/dingtalk.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/dingtalk.env"
  set +a
fi

: "${DINGTALK_WEBHOOK:?请设置 DINGTALK_WEBHOOK，或创建 dingtalk.env（见 dingtalk.env.example）}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

LOG_GLOB="${MINERU_LOG_GLOB:-/NAS_TYSK/02Data/大模型三水论文/run_shard*.log}"
INTERVAL="${INTERVAL:-1800}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/monitor_mineru_dingtalk.log}"
SCRIPT="$ROOT/monitor_mineru_dingtalk.py"
TITLE="${DINGTALK_TITLE:-MinerU 进度}"

pids() {
  pgrep -f '[m]onitor_mineru_dingtalk.py' || true
}

cmd_status() {
  local ps
  ps="$(pids)"
  if [[ -z "$ps" ]]; then
    echo 'monitor_mineru_dingtalk: not running'
    return 1
  fi
  echo 'monitor_mineru_dingtalk:'
  pgrep -af '[m]onitor_mineru_dingtalk.py' || true
  echo "log: $MONITOR_LOG"
  return 0
}

cmd_stop() {
  local ps
  ps="$(pids)"
  if [[ -z "$ps" ]]; then
    echo 'already stopped'
    return 0
  fi
  echo "stopping: $ps"
  # shellcheck disable=SC2086
  kill $ps 2>/dev/null || true
  sleep 1
  ps="$(pids)"
  if [[ -n "$ps" ]]; then
    # shellcheck disable=SC2086
    kill -9 $ps 2>/dev/null || true
  fi
  cmd_status || echo 'stopped'
}

cmd_once() {
  python -u "$SCRIPT" \
    --log-glob "$LOG_GLOB" \
    --title "$TITLE" \
    --once
}

cmd_start() {
  local ps
  ps="$(pids)"
  if [[ -n "$ps" ]]; then
    echo "already running: $ps"
    cmd_status
    return 0
  fi
  nohup python -u "$SCRIPT" \
    --log-glob "$LOG_GLOB" \
    --title "$TITLE" \
    --interval "$INTERVAL" \
    > "$MONITOR_LOG" 2>&1 &
  echo "PID=$!"
  sleep 1
  pgrep -af '[m]onitor_mineru_dingtalk.py' || true
  echo "---- last lines of $MONITOR_LOG ----"
  tail -n 20 "$MONITOR_LOG" || true
}

case "${1:-start}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  once)   cmd_once ;;
  *)
    echo "usage: $0 {start|status|stop|once}"
    exit 2
    ;;
esac
