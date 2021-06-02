#!/bin/bash
#SBATCH -J NN_TransUnet_Uni
#SBATCH --partition=gpu
#SBATCH -t 6:00:00
#SBATCH --mem=32GB
#SBATCH --nodelist=falcon4
# your code goes below
module load python-3.7.1
source ../../../paper_replication/TransUNet/NNproject/bin/activate
python test.py --dataset University --vit_name R50-ViT-B_16 --max_epochs 150  --crop 1 --add_cnn 1