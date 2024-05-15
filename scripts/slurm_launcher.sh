#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=cot-binary
#SBATCH --output=/checkpoint/%u/cot-binary/%a-%t.out
#SBATCH --error=/checkpoint/%u/cot-binary/%a-%t.err
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@meta.com

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=5:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --array=1-100


python /private/home/vivc/code/llm/cot/scripts/grid_run_exp1.py --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --config_filename exp1
