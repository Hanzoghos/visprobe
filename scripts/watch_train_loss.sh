#!/bin/bash
# Training loss watcher sentinel: reads the windowed average loss from the training log and kills the training process to trip the circuit breaker on anomalies.
# Rules (evaluated only after warmup, since loss is naturally high during warmup):
#   1) nan/inf appears           -> trip immediately
#   2) win_loss > 0.95           -> one-time warning (the random-guess line is ~1.06; nearing it is dangerous)
#   3) 3 consecutive monotonically rising points
#      with the last one > 0.80  -> trip (3B divergence trace: a 0.40 plateau followed by a rise all the way to 1.07)
# A normal "saved ->" print means training is complete; the sentinel exits.
# Usage: watch_train_loss.sh <training PID> <log path>
set -u
PID=$1; LOG=$2
RANDOM_GUESS=1.06
WARMUP_STEPS=750
last=""
rise=0
reported_high=0
while true; do
  if grep -qE "nan|inf" "$LOG" 2>/dev/null && grep -q "win" "$LOG" 2>/dev/null; then
    echo "[$(date '+%T')] FUSE: nan/inf found in log, killing $PID"; kill -9 $PID 2>/dev/null; exit 2
  fi
  if grep -q "saved ->" "$LOG" 2>/dev/null; then
    echo "[$(date '+%T')] Training ended normally; sentinel exiting"; exit 0
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "[$(date '+%T')] Training process $PID no longer exists (abnormal exit?); sentinel exiting"; exit 3
  fi
  # Extract the step and win loss from the latest training progress line
  line=$(grep "win.*_avg" "$LOG" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    step=$(echo "$line" | grep -oE "global [0-9]+" | grep -oE "[0-9]+")
    win=$(echo "$line" | grep -oE "win[0-9]+_avg [0-9.]+" | grep -oE "[0-9.]+$")
    if [ -n "$step" ] && [ "$step" -ge $WARMUP_STEPS ] 2>/dev/null; then
      if [ -n "$last" ]; then
        if awk "BEGIN{exit !($win > $last + 0.005)}"; then rise=$((rise+1)); else rise=0; fi
      fi
      if awk "BEGIN{exit !($win > 0.95)}" && [ $reported_high -eq 0 ]; then
        echo "[$(date '+%T')] WARN: step $step win_loss=$win is nearing the random-guess line $RANDOM_GUESS"; reported_high=1
      fi
      if [ $rise -ge 3 ] && awk "BEGIN{exit !($win > 0.80)}"; then
        echo "[$(date '+%T')] FUSE: step $step win_loss rose at 3 consecutive points to $win (suspected divergence), killing $PID"
        kill -9 $PID 2>/dev/null; exit 4
      fi
      last=$win
    fi
  fi
  sleep 60
done
