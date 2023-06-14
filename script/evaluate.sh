#!/bin/bash
#SBATCH --job-name=filament                  # Job name
#SBATCH --partition=gpu                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=1                           # Run 1 task
#SBATCH --cpus-per-task=8                    # Run 4 cores
#SBATCH --mem=24000                          # Job memory request in Megabytes
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --time=24:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/filament/logs/train_%j.log     # Standard output and error log

module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/FilamentDetection
python3 -i -m dfs.evaluation.detect --checkpoint_path '/gpfs/gpfs0/robert.jarolim/filament/unet_v3/checkpoint.pt' --data_path '/gpfs/gpfs0/robert.jarolim/data/filament/kso_img/*.jpg' --result_path '/gpfs/gpfs0/robert.jarolim/filament/unet_v3/kso'
