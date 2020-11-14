#!/bin/bash

#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00          # walltime
#SBATCH --mem=45G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=ii1g17@soton.ac.uk


module load conda
source activate gpu-iridis


cd $HOME/protein_ss_pred_phd


python protein_prediction.py
