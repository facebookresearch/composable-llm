#!/bin/bash

# Logging configuration
#SBATCH --job-name=difficulty_estimation
#SBATCH --output=/private/home/%u/logs/exp3/difficulty/logs/%a.log
#SBATCH --error=/private/home/%u/logs/exp3/difficulty/logs/%a.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=1-50
#SBATCH --time=1:00:00

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

python -m src.apps.gssm.difficulty --task-id $SLURM_ARRAY_TASK_ID --nb-tasks $SLURM_ARRAY_TASK_COUNT $PATH_TO_CODE_DIR/src/apps/gssm/configs/experiment3/difficulty.yaml 
