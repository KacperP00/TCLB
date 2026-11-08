#!/bin/bash
#SBATCH --job-name=TCLB_Anim
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgclb2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=kp/shallow_water/thesis/anim_%j.out
#SBATCH --error=kp/shallow_water/thesis/anim_%j.err

ulimit -l unlimited
set -e

cd /net/people/plgrid/plgkpiechocki/TCLB

echo "=== ŁADOWANIE MODUŁÓW ŚRODOWISKA ==="
module purge
module load GCC/11.3.0
module load GCCcore/11.3.0
module load libtirpc/1.3.2
module load OpenMPI/4.1.4
module load Python/3.10.4
module load R/4.2.3
module unload SciPy-bundle/2022.05

export USE_NUMEXPR=0
export USE_BOTTLENECK=0

# Opcjonalnie: bezgłowy tryb renderowania dla Matplotlib, żeby nie szukało serwera X11
export MPLBACKEND=Agg

SCRIPT_DIR="kp/shallow_water/thesis/topology_validation/animation"
SCALE=1.0
MODE="packet"  # lub "single"
OUTPUT_MP4="kp/shallow_water/thesis/topology_validation/animation/results/anim_sync_packet_1.0x.mp4"

echo "=== URUCHAMIANIE GENEROWANIA ANIMACJI ==="
python3 ${SCRIPT_DIR}/sync_anim.py ${SCALE} ${MODE} ${OUTPUT_MP4}

echo "=== ZAKOŃCZONO POMYŚLNIE ==="
date
