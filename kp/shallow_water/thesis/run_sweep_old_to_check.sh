#!/bin/bash
#SBATCH --job-name=SW_sweep
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=kp/shallow_water/thesis/sweep_%j.out
#SBATCH --error=kp/shallow_water/thesis/sweep_%j.err
#SBATCH --account=plgclb2025-gpu-a100

# Zabezpieczenie: Przejście do głównego katalogu (potrzebne dla ./p/run)
cd $HOME/TCLB

module load GCC/11.3.0 OpenMPI/4.1.4

# Załadowanie Pythona dostępnego na Athenie
source env/bin/activate

# Uruchomienie centrum dowodzenia
python kp/shallow_water/thesis/run_sweep.py
