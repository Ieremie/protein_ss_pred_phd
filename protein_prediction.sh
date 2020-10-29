#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=03:00:00          # walltime


# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=ii1g17@soton.ac.uk

cd $HOME

python protein_prediction.py
