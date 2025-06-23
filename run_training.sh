#!/bin/bash
conda activate colab
nohup python whisper_finetune_script.py > train.log 2>&1 &
