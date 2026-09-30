#!/bin/bash
# Generic single-run training job: one config in, one run directory out.
#   sbatch --gpus=<type>:1 scripts/train/train.sbatch <config.yaml> [--gpus 1] [key=value ...]
# This launcher is single-process only. Both the Slurm GPU allocation and optional `--gpus`
# argument must resolve to exactly one GPU.
# The #SBATCH lines carry resources only: account, partition, QoS and log paths are
# site-specific, and SLURM expands no variable inside a #SBATCH line, so pass them on the
# sbatch command line. scripts/repro.py does that from configs/local.yaml.
# Env: OCTTTA_REPO, OCTTTA_SCRATCH / OCTTTA_RUNS, OCTTTA_TRAIN_MODULE (default
# octtta.train), OCTTTA_ENV_ACTIVATE or OCTTTA_CONDA_SH + OCTTTA_CONDA_ENV.
#SBATCH --job-name=octtta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --requeue

set -euo pipefail

# WARNING: this section must stay at the top of the file. Its only purpose is to give
# SIGUSR1 an owner at EVERY moment; any line moved above it becomes a signal black hole.

#: Preemption is a NORMAL exit path and must report 0, or --requeue reschedules a job
PREEMPT_EXIT=0
#: Must be shorter than the partition's GraceTime, or SLURM sends SIGKILL first.
GRACE_SECONDS="${OCTTTA_GRACE_SECONDS:-25}"
READY_WAIT_SECONDS="${OCTTTA_READY_WAIT_SECONDS:-5}"
READY_ASSUME_SECONDS="${OCTTTA_READY_ASSUME_SECONDS:-180}"

PREEMPTED=0
PREEMPT_SIGNAL=""
WATCHDOG_PID=""
CHILD_RC=0
CHILD_STARTED_AT=0
PID=0
#: Set between `cmd &` and `PID=$!`, where the child exists but PID is still 0.
LAUNCH_IN_PROGRESS=0

# Contract with octtta/train.py: rank 0 touches OCTTTA_SIGNAL_READY_FILE as soon as its own handler is installed.
CHILD_READY_FILE="${TMPDIR:-/tmp}/octtta-usr1-ready.${SLURM_JOB_ID:-local}.$$"
rm -f "$CHILD_READY_FILE" 2>/dev/null || true
export OCTTTA_SIGNAL_READY_FILE="$CHILD_READY_FILE"
trap 'rm -f "$CHILD_READY_FILE" 2>/dev/null || true' EXIT

# Polls rather than sleeping once: a foreground `sleep` defers SIGTERM until it returns.
_start_grace_watchdog() {
  [ -z "$WATCHDOG_PID" ] || return 0
  (
    i=0
    while [ "$i" -lt "$GRACE_SECONDS" ]; do
      sleep 1
      kill -0 "$PID" 2>/dev/null || exit 0     # child is gone, nothing left to guard
      i=$(( i + 1 ))
    done
    echo "[slurm] grace period of ${GRACE_SECONDS}s expired, sending SIGKILL to $PID" >&2
    kill -KILL "$PID" 2>/dev/null || true
  ) &
  WATCHDOG_PID=$!
}

_await_child_ready() {
  [ -e "$CHILD_READY_FILE" ] && return 0
  local age=$(( $(date +%s) - CHILD_STARTED_AT ))
  if [ "$age" -ge "$READY_ASSUME_SECONDS" ]; then
    echo "[slurm] child has run ${age}s with no ready marker; assuming it does not implement the handshake and signalling now" >&2
    return 0
  fi
  echo "[slurm] child is only ${age}s old, waiting for its SIGUSR1 handler (<=${READY_WAIT_SECONDS}s)" >&2
  local ticks=0 max=$(( READY_WAIT_SECONDS * 5 ))
  while [ "$ticks" -lt "$max" ]; do
    if [ -e "$CHILD_READY_FILE" ]; then
      echo "[slurm] ready marker appeared after $(( ticks * 200 ))ms" >&2
      return 0
    fi
    kill -0 "$PID" 2>/dev/null || return 0     # child died on its own, nothing to wait for
    sleep 0.2
    ticks=$(( ticks + 1 ))
  done
  echo "[slurm] no ready marker after ${READY_WAIT_SECONDS}s; signalling anyway, which is still the better chance" >&2
}

_forward_signal() {
  local sig="$1"
  PREEMPTED=1
  PREEMPT_SIGNAL="$sig"

  if [ "${PID:-0}" -le 0 ]; then
    if [ "$LAUNCH_IN_PROGRESS" = "1" ]; then
      # Forked but not yet captured: exiting would orphan the child, so record and redeliver.
      echo "[slurm] SIG$sig at $(date -Is): caught between fork and PID capture, recording it for redelivery" >&2
      return 0
    fi
    echo "[slurm] SIG$sig at $(date -Is): no child started yet (setup window)" >&2
    echo "[slurm] nothing to checkpoint; exiting with preempted status $PREEMPT_EXIT so --requeue can reschedule" >&2
    exit "$PREEMPT_EXIT"
  fi

  echo "[slurm] SIG$sig at $(date -Is) -> pid $PID (grace ${GRACE_SECONDS}s)" >&2
  _start_grace_watchdog
  _await_child_ready
  kill -"$sig" "$PID" 2>/dev/null || true
}

install_signal_traps() {
  trap '_forward_signal USR1' USR1
  trap '_forward_signal TERM' TERM
  trap '_forward_signal INT'  INT
}

redeliver_pending_signal() {
  LAUNCH_IN_PROGRESS=0
  [ "$PREEMPTED" = "1" ] && [ -n "$PREEMPT_SIGNAL" ] || return 0
  echo "[slurm] redelivering SIG$PREEMPT_SIGNAL to the newly captured pid $PID" >&2
  _forward_signal "$PREEMPT_SIGNAL"
}

# A trap interrupting `wait` returns 128+signum WITHOUT reaping the child, so `kill -0`
# still succeeds and the loop continues until the child has really exited.
supervise_child() {
  local rc
  while :; do
    rc=0
    wait "$PID" || rc=$?
    if [ "$rc" -lt 128 ]; then break; fi
    kill -0 "$PID" 2>/dev/null || break
  done
  CHILD_RC="$rc"
  # Do not `wait` for the watchdog: it may sit in a sleep and burn most of the grace budget.
  if [ -n "$WATCHDOG_PID" ]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
}

final_exit_status() {
  if [ "$PREEMPTED" = "1" ]; then
    if [ "$CHILD_RC" -ne 0 ]; then
      echo "[slurm] child exited $CHILD_RC after SIG$PREEMPT_SIGNAL;" \
           "reporting $PREEMPT_EXIT to SLURM as a preemption (otherwise the job is marked FAILED)" >&2
    fi
    echo "$PREEMPT_EXIT"
  else
    echo "$CHILD_RC"
  fi
}

install_signal_traps          # first statement that does any work; all code below is covered

usage() {
  cat >&2 <<'EOU'
usage: sbatch --gpus=<type>:1 scripts/train/train.sbatch <config.yaml> [--gpus 1] [key=value ...]

  <config.yaml>   a config under configs/ (positional; see the note below)
  --gpus N        accepted only as 1; allocations and requests above one GPU are refused
  key=value       dotted octtta.config overrides, passed through unchanged

The config is passed POSITIONALLY to octtta.train. The positional form parses correctly
under both the old and the current argument parser, whereas `--config X a=b` once bound
config to "a=b".
EOU
}

CONFIG=""
GPU_REQUEST=""
PASSTHROUGH=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus)   GPU_REQUEST="${2:-}"; shift 2 || { usage; exit 2; } ;;
    --gpus=*) GPU_REQUEST="${1#--gpus=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [ -z "$CONFIG" ]; then CONFIG="$1"; else PASSTHROUGH+=("$1"); fi
      shift ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "!! no config given" >&2
  usage
  exit 2
fi
if [ -n "$GPU_REQUEST" ]; then
  case "$GPU_REQUEST" in ''|*[!0-9]*) echo "!! --gpus must be 1 in single-process mode, got '$GPU_REQUEST'" >&2; exit 2 ;; esac
  [ "$GPU_REQUEST" = "1" ] || { echo "!! --gpus $GPU_REQUEST is unsupported; this trainer runs one process on one GPU" >&2; exit 2; }
fi

# SLURM copies the batch script into its spool at SUBMIT time, so editing this never affects queued jobs.
REPO="${OCTTTA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -d "$REPO/octtta" ]; then
  echo "!! OCTTTA_REPO=$REPO has no octtta/ -- refusing to run somewhere that is not the repository" >&2
  exit 2
fi
echo "[train.sbatch] REPO=$REPO"

if [ ! -f "$REPO/$CONFIG" ] && [ ! -f "$CONFIG" ]; then
  echo "!! config not found: $CONFIG (cwd=$PWD, repo=$REPO)" >&2
  exit 2
fi

# Optional: neither variable set means the job runs the interpreter it inherited.
if [ -n "${OCTTTA_ENV_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "$OCTTTA_ENV_ACTIVATE"
elif [ -n "${OCTTTA_REPRO_ACTIVATE_CMD:-}" ]; then
  # A trusted local.yaml command supplied by repro.py (the user's own checkout).
  eval "$OCTTTA_REPRO_ACTIVATE_CMD"
elif [ -n "${OCTTTA_CONDA_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "$OCTTTA_CONDA_SH"
  conda activate "${OCTTTA_CONDA_ENV:?OCTTTA_CONDA_SH is set, so OCTTTA_CONDA_ENV must name the environment to activate}"
fi

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# cv2 opens its own thread pool in every dataloader worker, which oversubscribes the cores.
export OPENCV_NUM_THREADS=1

cd "$REPO"

_paths_value() { python -c "from octtta import paths; print(paths.$1)" 2>/dev/null || true; }
SCRATCH="${OCTTTA_SCRATCH:-$(_paths_value SCRATCH)}"
RUNS="${OCTTTA_RUNS:-$(_paths_value RUNS_DIR)}"
if [ -z "$SCRATCH" ] || [ -z "$RUNS" ]; then
  echo "!! could not resolve the scratch and runs roots through octtta/paths.py (is python" >&2
  echo "   on PATH? see OCTTTA_ENV_ACTIVATE above). Set OCTTTA_SCRATCH and OCTTTA_RUNS." >&2
  exit 2
fi
export OCTTTA_SCRATCH="$SCRATCH"
export OCTTTA_RUNS="$RUNS"
mkdir -p "${OCTTTA_LOG_DIR:-$RUNS/slurm}"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/xdg_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip_cache}"

# Training snapshots the resolved config at startup. A queued config edit is
# reported here so the operator can tell which version actually ran.
if [ -n "${OCTTTA_EXPECT_CONFIG_SHA256:-}" ]; then
  CONFIG_SHA256=$(python - "$CONFIG" <<'PY'
import hashlib
import json
import sys
from octtta.config import load_config
print(hashlib.sha256(json.dumps(load_config(sys.argv[1]), sort_keys=True).encode()).hexdigest())
PY
)
  if [ "$CONFIG_SHA256" != "$OCTTTA_EXPECT_CONFIG_SHA256" ]; then
    echo "[train.sbatch] config changed since submission; using the current file" >&2
  fi
fi

# Prefer Slurm's allocation metadata; CUDA_VISIBLE_DEVICES is the fallback in local jobs.
# The final nvidia-smi fallback is only for environments without those variables.
allocated_gpus() {
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then echo "$SLURM_GPUS_ON_NODE"; return; fi
  if [ -n "${SLURM_JOB_GPUS:-}" ]; then echo "${SLURM_JOB_GPUS}" | tr ',' '\n' | grep -c . ; return; fi
  if [ -n "${SLURM_STEP_GPUS:-}" ]; then echo "${SLURM_STEP_GPUS}" | tr ',' '\n' | grep -c . ; return; fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c . ; return; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count=$(nvidia-smi --list-gpus 2>/dev/null | grep -c . || true)
    echo "${count:-0}"
  else
    echo 0
  fi
}
NALLOC=$(allocated_gpus)
case "$NALLOC" in ''|*[!0-9]*)
  echo "!! could not parse the allocated GPU count '$NALLOC'; refusing an ambiguous launch" >&2
  exit 2 ;;
esac
if [ "$NALLOC" -gt 1 ]; then
  echo "!! this trainer supports one process and one GPU; this job was allocated $NALLOC GPUs" >&2
  echo "   submit with --gpus=<type>:1" >&2
  exit 2
fi
[ "$NALLOC" -ge 1 ] 2>/dev/null || NALLOC=1
NPROC=1

# Keep the single-process CPU-thread budget used by the one-GPU training path.
_cpus="${SLURM_CPUS_PER_TASK:-8}"
_omp=$(( _cpus / NPROC / 2 ))
[ "$_omp" -ge 1 ] || _omp=1
export OMP_NUM_THREADS="$_omp"

echo "=========================================================="
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "config     : $CONFIG"
echo "overrides  : ${PASSTHROUGH[*]:-<none>}"
echo "gpus       : nproc=$NPROC / allocated=$NALLOC (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "omp        : $OMP_NUM_THREADS  (cpus-per-task=$_cpus)"
echo "usr1 path  : direct to python"
echo "ready file : $CHILD_READY_FILE"
echo "restart    : ${SLURM_RESTART_COUNT:-0}"
echo "started    : $(date -Is)"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

TRAIN_MODULE="${OCTTTA_TRAIN_MODULE:-octtta.train}"

CHILD_STARTED_AT=$(date +%s)
LAUNCH_IN_PROGRESS=1
python -m "$TRAIN_MODULE" "$CONFIG" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} &
PID=$!
redeliver_pending_signal

supervise_child
RC=$(final_exit_status)

# Only wizard-submitted, fully completed runs perform this post-run pin check.
# A preempted job exits through the normal requeue path and must not be checked yet.
if [ "$RC" -eq 0 ] && [ -n "${OCTTTA_REPRO_CHECK_RUN:-}" ]; then
  python "$REPO/scripts/repro_checks.py" runs --only "$OCTTTA_REPRO_CHECK_RUN" ||
    echo "[train.sbatch] historical recipe check is advisory" >&2
fi

echo "finished   : $(date -Is)  child_rc=$CHILD_RC  preempted=$PREEMPTED  exit=$RC"
exit "$RC"
