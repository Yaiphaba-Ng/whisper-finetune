#!/bin/bash
pip install --upgrade --quiet pip
pip install --quiet -r requirements.txt
nohup python whisper_finetune_script.py > train.log 2>&1 &
