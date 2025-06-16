#!/bin/bash
nohup python whisper_finetune_script.py -g 1 > train.log 2>&1 &
