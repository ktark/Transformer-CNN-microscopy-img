#!/bin/bash
#SBATCH -J NN_TransUnet2
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --nodelist=falcon2
#SBATCH -t 16:00:00
# your code goes below
module load python-3.7.1
source ../../../paper_replication/TransUNet/NNproject/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --dataset University --vit_name R50-ViT-B_16 --max_epochs 150 --crop 1 --add_cnn 1
