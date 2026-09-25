#!/usr/bin/env bash
# Run the video pipeline while holding a systemd sleep/handle-lid-switch
# INHIBIT so the laptop cannot auto-suspend or lock mid-run — safe to leave
# running overnight even with the lid closed / no input for hours.
#
# On a desktop session where an X display exists, also silence the
# screensaver + DPMS so idle never pops a lock/login screen.
set -euo pipefail

display=":0"
if command -v xset >/dev/null 2>&1 && xset -display "$display" q >/dev/null 2>&1; then
    xset -display "$display" s off 2>/dev/null || true   # no screensaver blank
    xset -display "$display" -dpms 2>/dev/null || true   # no monitor power-down
fi

# Hold the inhibitor for the lifetime of the pipeline command.
systemd-inhibit --what=shutdown:sleep:handle-lid-switch:idle --why="auto-rendering videos" --mode=block -- \
    "$@"