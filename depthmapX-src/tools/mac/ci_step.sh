#!/bin/bash
# GitHub Actions helper: runs a command, shows its output, and when it fails publishes
# the last lines as an error annotation (annotations are readable without logging in,
# unlike the step logs).
#
#   tools/mac/ci_step.sh "<title>" <command> [args...]
set -uo pipefail

TITLE=$1
shift
LOG="$(mktemp)"
"$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    # annotation text: % \r \n must be escaped as %25 %0D %0A
    msg="$(tail -n 40 "$LOG" | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | awk '{printf "%s%%0A", $0}')"
    echo "::error title=$TITLE (exit code $rc)::$msg"
fi
rm -f "$LOG"
exit "$rc"
