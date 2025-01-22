#!/bin/bash

# Logging configuration
#SBATCH --job-name=data7
#SBATCH --output=/private/home/%u/logs/exp7/difficulty/logs/%a.log
#SBATCH --error=/private/home/%u/logs/exp7/difficulty/logs/%a.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --array=1-100
#SBATCH --time=10:00:00

# activate conda environment
eval "$(/private/home/vivc/miniconda/bin/conda shell.bash hook)"
conda activate /private/home/vivc/miniconda/envs/slm

# go to code directory
export PATH_TO_CODE_DIR=/private/home/vivc/code/composable-llm
cd $PATH_TO_CODE_DIR

# handle missing slurm variables
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
    SLURM_ARRAY_TASK_COUNT=1
fi

echo "Running task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"

python -m src.apps.gssm.data --task-id $SLURM_ARRAY_TASK_ID --nb-tasks $SLURM_ARRAY_TASK_COUNT $PATH_TO_CODE_DIR/src/apps/gssm/configs/experiment7/data.yaml 
