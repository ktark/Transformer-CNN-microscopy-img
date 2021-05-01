#!/bin/bash
#SBATCH -J hello_world
#SBATCH --partition=testing
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=500
# your code goes below
module load python/3.8.6
python -c 'print ("Hello world! Kaarel")'