#!/bin/sh
set -eu

state_dir=${XDG_RUNTIME_DIR:-/tmp}
uid=$(id -u)
state_file=${state_dir%/}/llama.cpp-perf-event-paranoid-$uid

if [ ! -f "$state_file" ]; then
    echo "No saved perf setting found for this user: $state_file" >&2
    exit 1
fi

original=$(cat "$state_file")
case "$original" in
    ''|*[!0-9-]*)
        echo "Invalid saved perf_event_paranoid value in $state_file" >&2
        exit 1
        ;;
    -* )
        digits=${original#-}
        case "$digits" in
            ''|*[!0-9]*)
                echo "Invalid saved perf_event_paranoid value in $state_file" >&2
                exit 1
                ;;
        esac
        ;;
esac

if sudo sysctl -w "kernel.perf_event_paranoid=$original"; then
    rm "$state_file"
    echo "Restored perf_event_paranoid to $original."
else
    echo "Could not restore the setting. The saved value remains at $state_file." >&2
    exit 1
fi
