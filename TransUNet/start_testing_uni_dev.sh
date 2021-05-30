#!/bin/bash
#SBATCH -J NN_TransUnet2
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
# your code goes below
module load python-3.7.1
source ../../../paper_replication/TransUNet/NNproject/bin/activate
python test.py --dataset University_dev --is_savenii --vit_name R50-ViT-B_16 --max_epochs 2000 --num_classes 2 --crop 1 --add_cnn 1