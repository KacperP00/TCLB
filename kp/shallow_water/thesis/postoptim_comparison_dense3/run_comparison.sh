#!/bin/bash
#SBATCH --job-name=TCLB_Thesis_Eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgclb2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=kp/shallow_water/thesis/eval_%j.out
#SBATCH --error=kp/shallow_water/thesis/eval_%j.err

ulimit -l unlimited
echo "running on: $(hostname)"
date

set -e # exit on error

# Ładowanie środowiska
module load GCC/11.3.0
module load GCCcore/11.3.0
module load libtirpc/1.3.2
module load OpenMPI/4.1.4
module load R/4.2.3
module load Python/3.10.4

export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

cd /net/people/plgrid/plgkpiechocki/TCLB

# Konfiguracja ścieżek
TCLB_EXEC="/net/people/plgrid/plgkpiechocki/TCLB/CLB/sw/main"
OUT_BASE="output/comparison"
SCRIPT_DIR="kp/shallow_water/thesis/postoptim_comparison"
# Główny skrypt leży wyżej, więc generator fali ciągłej poprawiamy na katalog thesis
WAVE_SCRIPT="kp/shallow_water/thesis/postoptim_comparison/singlewave_generator.py"

mkdir -p $OUT_BASE

echo "=== GENEROWANIE WYMUSZEŃ ==="
python3 $WAVE_SCRIPT 294.0
python3 $SCRIPT_DIR/gauss_fin.py

declare -A models=(
    ["clean_canal"]="$SCRIPT_DIR/clean_canal.xml"
    ["euve"]="$SCRIPT_DIR/euve_comparison.xml"
    ["topology"]="$SCRIPT_DIR/topology_basic.xml"
    ["empty_cav"]="$SCRIPT_DIR/empty_cavities.xml"
)

# --- 1. SYMULACJE I ANALIZA PACZEK FALOWYCH ---
echo "=== ETAP 1: PACZKI FALOWE (SZEROKOPASMOWE) ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_packet"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> Obliczenia GPU LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    sed -i 's|<CSV file="[^"]*"|<CSV file="'"$SCRIPT_DIR/inlet_packet.csv"'"|g' $temp_xml
    
    # Bezpośrednie uruchomienie przez MPI (zgodnie z ustawieniami SLURM ntasks=1)
    mpirun $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT dla paczki falowej..."
    python3 $SCRIPT_DIR/diagnostics_03.py $run_out_dir "packet_${model_name}"

    echo "-> Czyszczenie plików VTK dla $run_name..."
    rm ${run_out_dir}*.vti
done

# --- 2. SYMULACJE I ANALIZA FALI CIĄGŁEJ (294 LU) ---
echo "=== ETAP 2: FALA CIĄGŁA (REZYDUALNA - 294 LU) ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_single"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> Obliczenia GPU LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    # Wymuszenie ciągłe z katalogu wyższego
    sed -i 's|<CSV file="[^"]*"|<CSV file="kp/shallow_water/thesis/inlet_wave.csv"|g' $temp_xml
    
    mpirun $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT dla fali ciągłej..."
    python3 $SCRIPT_DIR/singlewave_diagnostics.py $run_out_dir "single_${model_name}"

    echo "-> Czyszczenie plików VTK dla $run_name..."
    rm ${run_out_dir}*.vti
done

echo "=== GENEROWANIE RAPORTU KOŃCOWEGO ==="
python3 $SCRIPT_DIR/report.py

echo "=== OBLICZENIA ZAKOŃCZONE POMYŚLNIE ==="
date
