#!/usr/bin/env bash
#
# Flowcat reproducible benchmark sweep. Runs INSIDE the container on the VM.
#
# Same WebSocket + μ-law load generator (io_harness load) drives two SUTs on the
# same box, with the load generator pinned to its own cores so it never steals
# cycles from the SUT:
#
#   flowcat-rust   — ONE process across all SUT cores (no GIL, tokio multi-thread)
#   pipecat-python — ONE process PER SUT core, SO_REUSEPORT (real multiprocess
#                    deployment; the kernel load-balances connections across them)
#
# Env knobs: SECS, FPS, SWEEP, LOADGEN_CORES.
set -uo pipefail

CORES=$(nproc)
LOADGEN_CORES=${LOADGEN_CORES:-4}
if [ "$CORES" -gt $((LOADGEN_CORES + 1)) ]; then
  SUT_CORES=$((CORES - LOADGEN_CORES))
else
  SUT_CORES=1
  LOADGEN_CORES=$((CORES - 1 > 0 ? CORES - 1 : 1))
fi
SUT_SET="0-$((SUT_CORES - 1))"
LOAD_SET="$SUT_CORES-$((CORES - 1))"
SECS=${SECS:-10}
FPS=${FPS:-50}
SWEEP=${SWEEP:-"10 25 50 100 250 500 1000 2000"}

# High concurrency needs lots of fds + connection backlog.
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true

echo "================ Flowcat benchmark ================"
echo "host_cores=$CORES  sut_cores=$SUT_CORES [$SUT_SET]  loadgen_cores=$LOADGEN_CORES [$LOAD_SET]"
echo "secs=$SECS  fps=$FPS  fd_limit=$(ulimit -n)"
echo "sweep (concurrent callers): $SWEEP"
python --version
echo "==================================================="

sweep() {
  local name=$1 url=$2
  echo
  echo "#### SUT = $name  ($url) ####"
  for c in $SWEEP; do
    taskset -c "$LOAD_SET" /app/io_harness load --url "$url" --conns "$c" --secs "$SECS" --fps "$FPS"
  done
}

# ---- flowcat-rust: one process, all SUT cores ----
taskset -c "$SUT_SET" /app/io_harness serve --addr 127.0.0.1:9099 &
RUST_PID=$!
sleep 1
sweep "flowcat-rust (1 proc / $SUT_CORES cores)" "ws://127.0.0.1:9099"
kill "$RUST_PID" 2>/dev/null || true
wait "$RUST_PID" 2>/dev/null || true

# ---- pipecat-python: one process per SUT core, SO_REUSEPORT ----
PY_PIDS=()
for c in $(seq 0 $((SUT_CORES - 1))); do
  taskset -c "$c" python /app/pipecat_sut.py --host 127.0.0.1 --port 9098 --reuse-port &
  PY_PIDS+=("$!")
  sleep 0.4  # stagger: launching N heavy pipecat imports at once thrashes mem/IO
done
# pipecat import takes several seconds — wait until the port actually accepts
# before sweeping (a fixed sleep raced the import and measured 0 throughput).
echo "waiting for pipecat (:9098) to accept connections..."
for i in $(seq 1 90); do
  python -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',9098))" 2>/dev/null && break
  sleep 1
done
sweep "pipecat-python ($SUT_CORES procs / $SUT_CORES cores)" "ws://127.0.0.1:9098"
for p in "${PY_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done

echo
echo "================ done ================"
echo "Compare p99 RTT + achieved throughput across the two SUTs at each caller count."
echo "Real-time voice needs a tight tail (p99); throughput<100% or climbing p99 = saturation."
