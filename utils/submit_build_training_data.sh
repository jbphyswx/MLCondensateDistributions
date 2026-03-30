#!/bin/bash
#SBATCH --job-name=cf_training_builder
#SBATCH --output=logs/cf_builder_%j.out
#SBATCH --error=logs/cf_builder_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --partition=compute

# Setup Julia environment
module load julia

# Ensure OhMyThreads has full access to the requested physical cores
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the working directory where the script expects to run
cd ${SLURM_SUBMIT_DIR}

echo "Starting massive Tabular extraction pipeline..."
echo "Allocated $JULIA_NUM_THREADS threads to OhMyThreads backend."

# Execute the orchestrator. If the job runs out of time or gets preempted, 
# re-running this exact script will rapidly skip existing .arrow files 
# and resume perfectly where it was terminated.
julia --project=.. utils/build_training_data.jl

echo "Pipeline executed successfully or completed all specified bounds."
