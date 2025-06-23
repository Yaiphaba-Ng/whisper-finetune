#!/bin/bash
nohup python whisper_finetune_script.py -T > train.log 2>&1 &
