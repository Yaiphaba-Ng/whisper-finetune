import os
import glob
import pandas as pd
from pathlib import Path
import re

# Hardcoded directories
EXCEL_DIR = r'datasets/lamzing'  # Directory containing Excel files
AUDIO_ROOT = r'datasets/lamzing/audio'  # Root directory containing audio files (may have subdirs)
OUTPUT_TSV = r'datasets/lamzing/combined.tsv'
NONPAIRS_TSV = r'datasets/lamzing/non_pairs.tsv'

# 1. Read all Excel files
excel_files = glob.glob(os.path.join(EXCEL_DIR, '*.xlsx'))
all_rows = []
for file in excel_files:
    df = pd.read_excel(file, usecols=['path', 'sentence'])
    all_rows.append(df)

# 2. Combine and sort by numeric part of filename in 'path'
combined = pd.concat(all_rows, ignore_index=True)

def extract_num(path):
    match = re.search(r'(\d+)', str(path))
    return int(match.group(1)) if match else float('inf')

combined = combined.dropna(subset=['path'])
combined['num'] = combined['path'].apply(lambda x: extract_num(Path(x).stem))
combined = combined.sort_values(by='num', ascending=True).reset_index(drop=True)


# 3. Search for each .mp3 file in AUDIO_ROOT (recursively)
def find_audio_file(filename):
    for root, dirs, files in os.walk(AUDIO_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None

valid_rows = []
non_pairs = []
for idx, row in combined.iterrows():
    path = str(row['path'])
    sentence = str(row['sentence']) if pd.notnull(row['sentence']) else ''
    audio_file = find_audio_file(Path(path).name)
    if audio_file and sentence.strip():
        valid_rows.append({'path': Path(audio_file).name, 'sentence': sentence})  # Only base filename
    else:
        non_pairs.append({'path': path, 'sentence': sentence})

# 4. Save valid pairs and non-pairs
df_valid = pd.DataFrame(valid_rows)
df_nonpairs = pd.DataFrame(non_pairs)
df_valid.to_csv(OUTPUT_TSV, sep='\t', index=False)
df_nonpairs.to_csv(NONPAIRS_TSV, sep='\t', index=False)

print(f"Done! {len(df_valid)} valid pairs, {len(df_nonpairs)} non-pairs.")
