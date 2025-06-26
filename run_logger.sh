#!/bin/bash
# Run system_metrics_logger.py in the background, detached

PYTHON_SCRIPT="system_metrics_logger.py"

nohup python "$PYTHON_SCRIPT" >/dev/null 2>&1 &
echo "System metrics logger started in background (PID $!). Output: ./system_metrics/<timestamp>.txt"
echo "To view the latest log, run:"
echo "  tail -f \$(ls -1t ./system_metrics/*.txt | head -n1)"
