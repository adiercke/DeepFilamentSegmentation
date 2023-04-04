#!/bin/bash
#SBATCH --job-name=filament                  # Job name
#SBATCH --partition=cpu                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=1                           # Run 1 task
#SBATCH --cpus-per-task=8                    # Run 4 cores
#SBATCH --mem=12000                          # Job memory request in Megabytes
#SBATCH --time=06:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/filament/logs/convert_%j.log     # Standard output and error log

module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/FilamentDetection
python3 -m dfs.data.gong_to_jpg
