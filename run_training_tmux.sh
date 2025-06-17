#!/bin/bash

SESSION_NAME="whisper_train"

# Start a new tmux session and run the raw python command
cmd="pip install --upgrade --quiet pip && pip install --quiet -r requirements.txt && python whisper_finetune_script.py -g 1 > train.log 2>&1"
tmux new-session -d -s $SESSION_NAME "$cmd"

echo "Started training in tmux session: $SESSION_NAME"
echo "To attach: tmux attach -t $SESSION_NAME"
