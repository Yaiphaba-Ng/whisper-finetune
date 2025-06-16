#!/bin/bash
pip install --upgrade --quiet pip
pip install --quiet -r requirements.txt
nohup python whisper_finetune_script.py -g 1 > train.log 2>&1 &
