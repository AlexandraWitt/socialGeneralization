#!/bin/bash

#SBATCH -J evol_Sim
#SBATCH -n 1                              # N cores
#SBATCH -N 1                              # All cores on one machine
#SBATCH -p cpu-short                       # Partition name                        # Memory 
#SBATCH --time=0-5:00            # Runtime in D-HH:MM
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=alexandra.witt@gmx.net   # Email to which notifications will be sent
#SBATCH --array=0-149


# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here
source $HOME/.bashrc
cd /mnt/qb/work/wu/awitt24/socialGeneralization/
conda activate socGen
python3 ./simulation/evoSim_Najar.py $SLURM_ARRAY_TASK_ID
conda deactivate 