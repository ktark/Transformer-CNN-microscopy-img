#!/bin/bash
#SBATCH -J NN_TransUnetUniNew
#SBATCH --partition=gpu
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
# your code goes below
module load python-3.7.1
source NNproject/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --dataset University --vit_name R50-ViT-B_16 --max_epochs 150