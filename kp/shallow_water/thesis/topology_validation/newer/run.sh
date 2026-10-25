#!/bin/bash
#SBATCH --job-name=TCLB_Universal_Eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgclb2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=kp/shallow_water/thesis/eval_univ_%j.out
#SBATCH --error=kp/shallow_water/thesis/eval_univ_%j.err

ulimit -l unlimited
set -e

# Pobranie mnoznika siatki z argumentu wywolania (np. sbatch run_universal.sh 2.0)
SCALE=${1:-1.0}
BASE_LAMBDA=${2:-324.0}

echo "=== URUCHAMIANIE POTOKU WALIDACYJNEGO DLA SIATKI ${SCALE}x (lambda_1x=${BASE_LAMBDA} LU) ==="
date

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

cd /net/people/plgrid/plgkpiechocki/TCLB

TCLB_EXEC="/net/people/plgrid/plgkpiechocki/TCLB/CLB/sw/main"
SCRIPT_DIR="kp/shallow_water/thesis/topology_validation/newer"
THESIS_DIR="kp/shallow_water/thesis"
OUT_BASE="output/validation_scale_${SCALE}x"
RES_DIR="${THESIS_DIR}/wyniki_fit_${SCALE}x"

# Dlugosc rekordu (iteracje przy skali 1.0x): paczka 65000 = jak w optymalizacji, fala ciagla 71000 (dluzsze okno stanu ustalonego)
PACKET_ITERS=65000
SINGLE_ITERS=71000

PACKET_CSV="${THESIS_DIR}/inlet_packet_${SCALE}x.csv"
SINGLE_CSV="${THESIS_DIR}/inlet_wave_${SCALE}x.csv"
BASE_STL="${THESIS_DIR}/basic_mesh_topology_00.stl"
SCALED_STL="${THESIS_DIR}/topology_${SCALE}x.stl"

# Wygenerowane XML-e zostaja do wgladu (male pliki), nie sa usuwane
XML_DIR="${RES_DIR}/xml"
mkdir -p $OUT_BASE $RES_DIR $XML_DIR

echo "=== 1. SKALOWANIE STL I GENEROWANIE WYMUSZEN (${SCALE}x) ==="
python3 $SCRIPT_DIR/stl_rescale.py $SCALE $BASE_STL $SCALED_STL
python3 $SCRIPT_DIR/gen_gauss.py $SCALE $PACKET_CSV $PACKET_ITERS
python3 $SCRIPT_DIR/gen_singlewave.py $SCALE $BASE_LAMBDA $SINGLE_CSV $SINGLE_ITERS

declare -A templates=(
    ["clean_canal"]="$SCRIPT_DIR/clean_canal.xml.tpl"
    ["euve"]="$SCRIPT_DIR/euve.xml.tpl"
    ["topology"]="$SCRIPT_DIR/topology.xml.tpl"
    ["empty_cav"]="$SCRIPT_DIR/empty_cavities.xml.tpl"
)

echo "=== 2. ETAP 1: PACZKI FALOWE (${SCALE}x) ==="
for model_name in "${!templates[@]}"; do
    tpl_source="${templates[$model_name]}"
    run_name="${model_name}_packet_${SCALE}x"
    run_out_dir="${OUT_BASE}/${run_name}/"
    mkdir -p $run_out_dir
    temp_xml="${XML_DIR}/${run_name}.xml"

    python3 $SCRIPT_DIR/prepare_xmls.py $SCALE $tpl_source $temp_xml $run_out_dir $PACKET_CSV $SCALED_STL $PACKET_ITERS
    mpirun $TCLB_EXEC $temp_xml

    python3 $SCRIPT_DIR/diagnostics.py $run_out_dir "packet_${model_name}" $SCALE $RES_DIR
    rm ${run_out_dir}*.vti
done

echo "=== 3. ETAP 2: FALA CIAGLA (${SCALE}x) ==="
for model_name in "${!templates[@]}"; do
    tpl_source="${templates[$model_name]}"
    run_name="${model_name}_single_${SCALE}x"
    run_out_dir="${OUT_BASE}/${run_name}/"
    mkdir -p $run_out_dir
    temp_xml="${XML_DIR}/${run_name}.xml"

    python3 $SCRIPT_DIR/prepare_xmls.py $SCALE $tpl_source $temp_xml $run_out_dir $SINGLE_CSV $SCALED_STL $SINGLE_ITERS
    mpirun $TCLB_EXEC $temp_xml

    python3 $SCRIPT_DIR/diagnostics.py $run_out_dir "single_${model_name}" $SCALE $RES_DIR $BASE_LAMBDA
    rm ${run_out_dir}*.vti
done

echo "=== 4. GENEROWANIE RAPORTU ==="
python3 $SCRIPT_DIR/report.py $SCALE
date