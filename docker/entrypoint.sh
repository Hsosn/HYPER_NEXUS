#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Hyper-Nexus container entrypoint.
#
# Supervises the FastAPI web server, the Celery worker, and the
# Celery beat scheduler. Pass any subset of {web, worker, beat}
# as CMD arguments to run only what you need (e.g. scale workers
# separately in a docker swarm).
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd /app

# Make sure a .env exists. We do not fail if it does not — the
# application has its own defaults and may run with just an LLM
# key set as a real env-var.
if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    echo "[entrypoint] no .env present — copied .env.example. Edit /app/.env to add LLM keys."
fi

# Default to running everything if the user gave no arguments.
if [[ $# -eq 0 ]]; then
    set -- web worker beat
fi

pids=()

# Start a process and record its PID. If it dies, we want the whole
# container to die so the orchestrator can restart it.
run() {
    local name="$1"; shift
    echo "[entrypoint] starting $name: $*"
    "$@" &
    pids+=($!)
}

trap 'echo "[entrypoint] shutting down"; kill "${pids[@]}" 2>/dev/null || true; wait' SIGTERM SIGINT

for svc in "$@"; do
    case "$svc" in
        web)
            run web \
                uvicorn nexus.api.server:app \
                    --host "${NEXUS_HOST}" --port "${NEXUS_PORT}" \
                    --proxy-headers --forwarded-allow-ips='*'
            ;;
        worker)
            run worker \
                celery -A nexus.celery_app worker -l info --concurrency=2
            ;;
        beat)
            run beat \
                celery -A nexus.celery_app beat -l info
            ;;
        *)
            echo "[entrypoint] unknown service: $svc" >&2
            exit 1
            ;;
    esac
done

# Wait for the first child to exit, then bring the rest down.
wait -n "${pids[@]}"
echo "[entrypoint] a supervised process exited — stopping the others"
kill "${pids[@]}" 2>/dev/null || true
wait
exit 0
