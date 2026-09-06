#!/bin/bash
# Zdefiniowanie nazwy zadania w kolejce
#SBATCH --job-name=SW_Opt
# Alokacja zasobow sprzetowych: 1 wezel, 1 karta A100, 60GB RAM
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
# Zabezpieczenie limitu czasu (format GG:MM:SS)
#SBATCH --time=24:00:00
# Powiazanie zadania z aktywnym grantem
#SBATCH --account=plgclb2026-gpu-a100
# Zrzut logow standardowych i bledow do plikow
#SBATCH --output=kp/shallow_water/thesis/opt_%j.out
#SBATCH --error=kp/shallow_water/thesis/opt_%j.err

# Przejscie do glownego katalogu struktury
cd $HOME/TCLB

# Uruchomienie procesu optymalizacji solverem TCLB
CLB/sw/main kp/shallow_water/thesis/optimization.xml