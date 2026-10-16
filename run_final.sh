#!/bin/bash
#SBATCH --job-name=TCLB:kp/shallow_water/thesis/optimization_06.xml
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgclb2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=kp/shallow_water/thesis/opt_%j.out
#SBATCH --error=kp/shallow_water/thesis/opt_%j.err

ulimit -l unlimited
echo "running on:"
hostname

date

set -e # exit on error
module load GCC/11.3.0
module load GCCcore/11.3.0
module load libtirpc/1.3.2
module load OpenMPI/4.1.4
module load R/4.2.3

cd /net/people/plgrid/plgkpiechocki/TCLB
mpirun   /net/people/plgrid/plgkpiechocki/TCLB/CLB/sw/main kp/shallow_water/thesis/optimization_06.xml 
