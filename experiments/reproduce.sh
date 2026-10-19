#!/usr/bin/env bash
# Reproduce the experiments of the SofaGym paper (Schegg, Ménager et al., Soft Robotics 2023).
#
# Resumable: each run saves its model and logs after every epoch, rerunning this script resumes the unfinished runs and
# skips the finished ones. A run whose inputs changed (the code of its env, of SofaGym, of the benchmark) restarts from
# scratch, its previous results are moved to <run>.old-<date>.
#
# Round robin: a benchmark run trains a chunk of 50 epochs per launch, the free CPUs go to the unfinished run with the
# fewest chunks, so that all the runs progress together.
#
# Usage (from the repository, in the Python env of SofaGym with SOFA_ROOT and PYTHONPATH set, see README.md):
#     experiments/reproduce.sh
#     SEEDS="0 1" ENVS="cartstem-v0" ALGOS=PPO experiments/reproduce.sh
#
# Variables: OUT (results, default experiments/results), SEEDS (default "0 1 2 3 4 5", the 6 seeds of the paper), ENVS,
# ALGOS (default all: SAC, PPO, OPD), CPUS (processes at once: a SAC run has 1, a PPO run 5 with its 4 envs; ~2.3 GB of RAM each;
# default: the CPUs, or less if the available memory is short), RETRIES (attempts per run, default 3), PYTHON (default
# python).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${OUT:-$ROOT/experiments/results}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
# Up to ~2.3 GB per process (a new SOFA scene per episode leaks a little, the runs restart every 50 epochs), and 6 GB
# left for the rest of the machine.
MEMORY_CPUS=$(awk '/MemAvailable/ {print int(($2 - 3000000) / 2000000)}' /proc/meminfo)
CPUS=${CPUS:-$(( $(nproc) < MEMORY_CPUS ? $(nproc) : MEMORY_CPUS ))}
RETRIES=${RETRIES:-3}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# One thread per run for torch, which would otherwise use all the CPUs in each of the parallel runs.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# RL baseline of the paper (section 5.1, figure 2): env id, its scene directory, its algorithms (SAC needs continuous
# actions).
BENCHMARK=(
    "cartstemcontact-v0 CartStemContact SAC PPO"
    "cartstem-v0 CartStem PPO"
    "catchtheobject-v0 CatchTheObject SAC PPO"
)

inputs_hash() {
    # The code a run depends on: its env, the core of SofaGym, the agents' helpers and its script (default the benchmark).
    # Not the Results/ of the envs: data they write while running (Diamond, CTR), file names with spaces.
    find "sofagym/envs/$1" sofagym/*.py agents/utils.py "${2:-experiments/benchmark.py}" -path '*/Results' -prune \
        -o \( -name '*.py' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

finished() {
    # finished <dir> <hash>: done with the same inputs, or failed RETRIES times in this invocation of the script.
    [[ -f $1/FAILED ]] || [[ -f $1/DONE && $(cat "$1/DONE") == "$2" ]]
}

run_stage() {
    # run_stage <dir> <hash> <command...>: one chunk of the run (all of it for planning), restart when the inputs changed.
    local dir=$1 hash=$2
    shift 2
    if [[ -f $dir/INPUTS && $(cat "$dir/INPUTS") != "$hash" ]]; then
        echo "[inputs changed] $(basename "$dir"): restart, previous results in $dir.old-$(date +%F-%H%M%S)"
        mv "$dir" "$dir.old-$(date +%F-%H%M%S)"
    fi
    mkdir -p "$dir"
    echo "$hash" > "$dir/INPUTS"
    local attempt=1 marker
    echo "[run] $(basename "$dir")"
    while (( attempt <= RETRIES )); do
        # The marker of this invocation: done, continue (the run exits regularly to free memory), or none (failure).
        marker=$("$@" 2>&1 | tee -a "$dir/output.log" | grep -E '^[A-Z]+_(DONE|CONTINUE)$' | tail -1) || true
        case $marker in
            *_DONE) echo "$hash" > "$dir/DONE"; echo "[done] $(basename "$dir")"; return 0 ;;
            *_CONTINUE) echo "[chunk done] $(basename "$dir")"; return 0 ;;
            *) attempt=$((attempt + 1)); (( attempt <= RETRIES )) && echo "[retry] $(basename "$dir") (attempt $attempt/$RETRIES)" ;;
        esac
    done
    touch "$dir/FAILED"
    echo "[failed] $(basename "$dir"), see $dir/output.log"
    return 1
}

declare -A PROCESSES  # running runs: pid -> number of processes
running_processes() {
    local n=0 pid
    for pid in "${!PROCESSES[@]}"; do
        kill -0 "$pid" 2> /dev/null && n=$((n + PROCESSES[$pid]))
    done
    echo "$n"
}

STAGES=()  # "<processes> <epochs per launch> <dir> <inputs hash> <command>"
for entry in "${BENCHMARK[@]}"; do
    read -r env_id scene algos <<< "$entry"
    [[ -n ${ENVS:-} && " $ENVS " != *" $env_id "* ]] && continue
    hash=$(inputs_hash "$scene")
    for algo in $algos; do
        [[ -n ${ALGOS:-} && " $ALGOS " != *" $algo "* ]] && continue
        for seed in $SEEDS; do
            processes=$([[ $algo == PPO ]] && echo 5 || echo 1)
            # CatchTheObject leaks ~6.5 MB per episode (~25 times the others): shorter chunks to stay under ~2.3 GB.
            chunk=$([[ $env_id == catchtheobject-v0 ]] && echo 10 || echo 50)
            STAGES+=("$processes $chunk $OUT/benchmark/${algo}_${env_id}_${seed} $hash $PYTHON experiments/benchmark.py $env_id $algo $seed --out $OUT/benchmark --epochs $chunk")
        done
    done
done
# Planning of the paper (sections 5.3 and 5.4): OPD on random targets (Diamond, ConcentricTubeRobot) and on the goals of
# the env (Trunk's 2, SimpleMaze's 4 corners). Budget per real step: 4 expansions of all the actions (not given in the
# paper), 16 for ConcentricTubeRobot (see README.md). env id, its scene directory, its targets, its budget.
PLANNING=(
    "diamondrobot-v0 Diamond 10 32"
    "concentrictuberobot-v0 CTR 10 192"
    "trunk-v0 Trunk 2 64"
    "simple_maze-v0 SimpleMaze 4 16"
)
for entry in "${PLANNING[@]}"; do
    read -r env_id scene targets budget <<< "$entry"
    [[ -n ${ENVS:-} && " $ENVS " != *" $env_id "* ]] && continue
    [[ -n ${ALGOS:-} && " $ALGOS " != *" OPD "* ]] && continue
    hash=$(inputs_hash "$scene" experiments/planning.py)
    for target in $(seq 0 $((targets - 1))); do
        # The clients of the planner simulate one transition at a time.
        STAGES+=("2 50 $OUT/planning/OPD_${env_id}_${target} $hash $PYTHON experiments/planning.py $env_id $target --budget $budget --out $OUT/planning")
    done
done

# Round robin over the stages: launch an unfinished, not running stage with the fewest epochs that fits in the free
# CPUs (a run of the next round waits for this round to be launched, so that the PPO runs are not starved).
rm -f "$OUT"/benchmark/*/FAILED "$OUT"/planning/*/FAILED
declare -A EPOCHS ACTIVE  # dir -> epochs done or launched, dir -> pid of its running chunk
for stage in "${STAGES[@]}"; do
    # The epochs done before this invocation, so that a restart of the script does not favour the first stages.
    read -r _ _ dir _ <<< "$stage"
    epochs=$(grep -so 'epoch [0-9]*/' "$dir/output.log" | tail -1 | tr -dc 0-9 || true)
    EPOCHS[$dir]=${epochs:-0}
done
while :; do
    next="" pending=0 least="" free=$(( CPUS - $(running_processes) ))
    for stage in "${STAGES[@]}"; do
        read -r processes _ dir hash _ <<< "$stage"
        finished "$dir" "$hash" && continue
        pending=1
        [[ -n ${ACTIVE[$dir]:-} ]] && kill -0 "${ACTIVE[$dir]}" 2> /dev/null && continue
        epochs=${EPOCHS[$dir]}
        if [[ -z $least ]] || (( epochs < least )); then
            least=$epochs next=""
        fi
        # Alone, a stage runs even with more processes than CPUS.
        if [[ -z $next ]] && (( epochs == least && (processes <= free || free == CPUS) )); then
            next=$stage
        fi
    done
    (( pending )) || break
    if [[ -z $next ]]; then
        wait -n || true
        continue
    fi
    read -r processes chunk dir hash command <<< "$next"
    # shellcheck disable=SC2086  # the command is split into its words
    run_stage "$dir" "$hash" $command &
    PROCESSES[$!]=$processes ACTIVE[$dir]=$! EPOCHS[$dir]=$(( EPOCHS[$dir] + chunk ))
done
wait
failed=$(find "$OUT/benchmark" "$OUT/planning" -maxdepth 1 -mindepth 1 -type d ! -name '*.old-*' ! -exec test -f '{}/DONE' \; \
    -print 2> /dev/null | wc -l)

"$PYTHON" experiments/plot_benchmark.py "$OUT/benchmark"
"$PYTHON" experiments/plot_benchmark.py "$OUT/benchmark" --final
"$PYTHON" experiments/summarize_planning.py "$OUT/planning"
"$PYTHON" experiments/video.py "$OUT"

# What is not reproduced yet, and why: experiments/README.md.

if (( failed )); then
    echo "$failed runs not done, rerun this script to resume them"
    exit 1
fi
