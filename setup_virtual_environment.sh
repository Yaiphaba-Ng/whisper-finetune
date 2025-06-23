#!/bin/bash
conda create -n colab python=3.11 --yes
conda activate colab
conda install conda-forge::google-colab --yes
pip install --upgrade --quiet pip
pip install --quiet -r requirements.txt
