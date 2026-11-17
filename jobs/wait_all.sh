# Sourced by the job files. wait_all waits for every pid in the array `pids`; when any run
# fails it stops the others (they resume from checkpoints on resubmit) and exits with the
# failing status. Each finished pid is collected with `wait "$pid"`: bash 4.4 (SCC) has no
# `wait -n -p`, and a child that exits while another is being handled is not reported by
# the next `wait -n`, so `wait -n` only serves to sleep until something finishes.
wait_all() {
  local status=0 rc p alive
  while :; do
    alive=()
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        alive+=("$p")
      else
        rc=0; wait "$p" || rc=$?
        if [ "$rc" -ne 0 ]; then echo "run pid $p exited with status $rc"; status=$rc; fi
      fi
    done
    pids=("${alive[@]}")
    if [ "$status" -ne 0 ]; then
      if [ "${#pids[@]}" -gt 0 ]; then
        echo "a run failed (exit $status); stopping the other runs"
        for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done
        wait || true
      fi
      exit "$status"
    fi
    [ "${#pids[@]}" -eq 0 ] && return 0
    wait -n || true
  done
}
