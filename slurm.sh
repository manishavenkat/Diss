#!/bin/bash
#
#SBATCH --partition=gpu 
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool2
#SBATCH --job-name=classification_gs_sample
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# module load python/3.10.8-gpu
# module load pytorch/1.13.1-gpu

export HF_HOME="/work/tc062/tc062/manishav/huggingface_cache"
# export HF_HOME="/scratch/space1/tc062/manishav/gigaspeech/work/tc062/tc062/manishav/huggingface_cache"
export TRANSFORMERS_CACHE="/work/tc062/tc062/manishav/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/work/tc062/tc062/manishav/huggingface_cache/datasets"
# export HF_DATASETS_CACHE="/scratch/space1/tc062/manishav/gigaspeech/work/tc062/tc062/manishav/huggingface_cache/datasets"

export MPLCONFIGDIR="/work/tc062/tc062/manishav/.config/matplotlib"

source /work/tc062/tc062/manishav/.venv/diss/bin/activate 
# python -m pip install "numpy<2"
srun python /work/tc062/tc062/manishav/Diss/explore_gs.py
