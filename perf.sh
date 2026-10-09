#!/usr/bin/env bash
set -euo pipefail

KNOB=/proc/sys/kernel/perf_event_paranoid
STATE=${XDG_RUNTIME_DIR:-/tmp}/perf-paranoid.prev
ENABLED_VALUE=1

case "${1:-status}" in
    enable)
        [ -e "$STATE" ] || cat "$KNOB" > "$STATE"
        sudo sysctl -q -w kernel.perf_event_paranoid="$ENABLED_VALUE"
        echo "perf enabled (perf_event_paranoid=$(cat "$KNOB"), previous=$(cat "$STATE"))"
        ;;
    disable)
        prev=$(cat "$STATE" 2>/dev/null || echo 3)
        sudo sysctl -q -w kernel.perf_event_paranoid="$prev"
        rm -f "$STATE"
        echo "perf disabled (perf_event_paranoid=$(cat "$KNOB"))"
        ;;
    status)
        echo "perf_event_paranoid=$(cat "$KNOB")"
        ;;
    *)
        echo "usage: $0 enable|disable|status" >&2
        exit 2
        ;;
esac
