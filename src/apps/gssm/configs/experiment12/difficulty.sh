#!/bin/bash

# Logging configuration
#SBATCH --job-name=difficulty_estimation
#SBATCH --output=/private/home/%u/logs/exp12/difficulty/logs/%a.log
#SBATCH --error=/private/home/%u/logs/exp12/difficulty/logs/%a.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=1-100
#SBATCH --time=1:00:00

# go to code directory
cd $CODE_DIR

# handle missing slurm variables
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
    SLURM_ARRAY_TASK_COUNT=1
fi

echo "Running task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"

export FILENAME=onfly
export BSZ=128
export NBATCHES=50

python -m src.apps.gssm.difficulty --task-id $SLURM_ARRAY_TASK_ID --nb-tasks $SLURM_ARRAY_TASK_COUNT --bsz $BSZ --n-batches $NBATCHES $CODE_DIR/src/apps/gssm/configs/experiment12/$FILENAME.yaml 
