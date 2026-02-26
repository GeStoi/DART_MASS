#!/bin/bash
set -e

# Start Xvfb (virtual framebuffer) for headless OpenGL rendering
# This is needed because the render module uses GLUT which requires a display
if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":99" ]; then
    echo "[MASS] Starting Xvfb virtual display on :99..."
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
    export DISPLAY=:99
    # Wait for Xvfb to start
    sleep 1
fi

# If DISPLAY is set to something else (e.g., X11 forwarding), use that instead
echo "[MASS] Using DISPLAY=$DISPLAY"
echo "[MASS] Working directory: $(pwd)"
echo "[MASS] MASS environment ready."
echo ""

# Execute the command passed to docker run
exec "$@"
