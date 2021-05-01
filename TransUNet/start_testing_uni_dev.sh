#!/bin/bash
#SBATCH -J NN_TransUnet2
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
# your code goes below
module load python-3.7.1
source NNproject/bin/activate
python test.py --dataset University_dev --is_savenii --vit_name R50-ViT-B_16