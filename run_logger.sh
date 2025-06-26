#!/bin/bash
# Run system_metrics_logger.py in the background, detached

LOGFILE="system_metrics_logger.out"
PYTHON_SCRIPT="system_metrics_logger.py"

nohup python "$PYTHON_SCRIPT" > "$LOGFILE" 2>&1 &
echo "System metrics logger started in background (PID $!). Output: $LOGFILE"
