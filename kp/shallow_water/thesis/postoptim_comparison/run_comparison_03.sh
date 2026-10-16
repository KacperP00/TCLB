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
#SBATCH --time=02:00:00
#SBATCH --output=kp/shallow_water/thesis/eval_new_%j.out
#SBATCH --error=kp/shallow_water/thesis/eval_new_%j.err

ulimit -l unlimited
echo "running on: $(hostname)"
date

set -e # exit on error

# Czyste ładowanie podstawowych bibliotek
module purge
module load GCC/11.3.0
module load GCCcore/11.3.0
module load libtirpc/1.3.2
module load OpenMPI/4.1.4
module load Python/3.10.4

# Ładujemy główny kombajn, żeby TCLB dostał HDF5, NLopt i całą resztę...
module load R/4.2.3

# ...a następnie chirurgicznie usuwamy stare biblioteki Pythona, które R pociągnął za sobą
module unload SciPy-bundle/2022.05

# Zabezpieczenie dla Pandas
export USE_NUMEXPR=0
export USE_BOTTLENECK=0

cd /net/people/plgrid/plgkpiechocki/TCLB

TCLB_EXEC="/net/people/plgrid/plgkpiechocki/TCLB/CLB/sw/main"
OUT_BASE="output/comparison"
SCRIPT_DIR="kp/shallow_water/thesis/postoptim_comparison"
CSV_DIR="kp/shallow_water/thesis"

mkdir -p $OUT_BASE

echo "=== GENEROWANIE WYMUSZEŃ ==="
python3 $SCRIPT_DIR/singlewave_generator.py 324.0
python3 $SCRIPT_DIR/gauss_fin.py

declare -A models=(
    ["clean_canal"]="$SCRIPT_DIR/clean_canal.xml"
    ["euve"]="$SCRIPT_DIR/euve_comparison.xml"
    ["topology"]="$SCRIPT_DIR/topology_basic.xml"
    ["empty_cav"]="$SCRIPT_DIR/empty_cavities.xml"
)

# --- 1. SYMULACJE I ANALIZA PACZEK FALOWYCH ---
echo "=== ETAP 1: PACZKI FALOWE ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_packet"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    # Poprawiona ścieżka do CSV z paczką (kieruje do folderu nadrzędnego)
    sed -i 's|<CSV file="[^"]*"|<CSV file="'"$CSV_DIR/inlet_packet.csv"'"|g' $temp_xml
    
    mpirun $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT: $run_name"
    python3 $SCRIPT_DIR/diagnostics_04.py $run_out_dir "packet_${model_name}"

    echo "-> Czyszczenie: $run_name"
    rm ${run_out_dir}*.vti
done

# --- 2. SYMULACJE I ANALIZA FALI CIĄGŁEJ ---
echo "=== ETAP 2: FALA CIĄGŁA ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_single"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    # Poprawiona ścieżka do CSV z falą ciągłą (kieruje do folderu nadrzędnego)
    sed -i 's|<CSV file="[^"]*"|<CSV file="'"$CSV_DIR/inlet_wave.csv"'"|g' $temp_xml
    
    mpirun $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT: $run_name"
    python3 $SCRIPT_DIR/singlewave_diagnostics_02.py $run_out_dir "single_${model_name}"

    echo "-> Czyszczenie: $run_name"
    rm ${run_out_dir}*.vti
done

echo "=== GENEROWANIE RAPORTU ==="
python3 $SCRIPT_DIR/report.py
date
