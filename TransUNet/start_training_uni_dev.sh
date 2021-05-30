#!/bin/bash
#SBATCH -J NN_TransUnet3
#SBATCH --partition=gpu
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:10:00
# your code goes below
module load python-3.7.1
source NNproject/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --dataset University_dev --vit_name R50-ViT-B_16 --max_epochs 10