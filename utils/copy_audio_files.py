import os
import shutil
import csv

# Set the path to your TSV file here
TSV_PATH = r"evals/checkpoints/whisper-medium-as/checkpoint-1000/eval_20250619_135140.tsv"
AUDIO_SRC_DIR = r"evals/as_test"
AUDIO_DST_DIR = os.path.join(os.path.dirname(TSV_PATH), 'audio')

os.makedirs(AUDIO_DST_DIR, exist_ok=True)

with open(TSV_PATH, 'r', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        filename = row['path']
        src = os.path.join(AUDIO_SRC_DIR, filename)
        dst = os.path.join(AUDIO_DST_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {filename}")
        else:
            print(f"Not found: {filename}")
