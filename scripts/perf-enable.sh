#!/bin/sh
set -eu

state_dir=${XDG_RUNTIME_DIR:-/tmp}
uid=$(id -u)
state_file=${state_dir%/}/llama.cpp-perf-event-paranoid-$uid

if [ ! -d "$state_dir" ]; then
    echo "Runtime directory does not exist: $state_dir" >&2
    exit 1
fi

if [ -e "$state_file" ] || [ -L "$state_file" ]; then
    echo "A saved perf setting already exists: $state_file" >&2
    echo "Run scripts/perf-disable.sh before enabling it again." >&2
    exit 1
fi

original=$(cat /proc/sys/kernel/perf_event_paranoid)
if ! (umask 077; set -C; printf '%s\n' "$original" > "$state_file") 2>/dev/null; then
    echo "Could not save the current perf_event_paranoid value." >&2
    exit 1
fi

if sudo sysctl -w kernel.perf_event_paranoid=0; then
    echo "Enabled perf CPU counters. Original value $original saved for this user."
else
    echo "Could not change the setting. The saved value remains at $state_file." >&2
    echo "Run scripts/perf-disable.sh to restore it if needed." >&2
    exit 1
fi
