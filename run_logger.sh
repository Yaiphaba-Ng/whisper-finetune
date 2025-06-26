#!/bin/bash
# Run system_metrics_logger.py in the background, detached

PYTHON_SCRIPT="system_metrics_logger.py"

nohup python "$PYTHON_SCRIPT" &
echo "System metrics logger started in background (PID $!). Output: ./system_metrics/<timestamp>.txt"
