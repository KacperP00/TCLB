#!/bin/bash
#SBATCH --job-name=TCLB_Sweep_Anim
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgclb2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=kp/shallow_water/thesis/sweep_%j.out
#SBATCH --error=kp/shallow_water/thesis/sweep_%j.err

ulimit -l unlimited
set -e

SCALE=${1:-1.0}
ITERS=71000

module purge
module load GCC/11.3.0 GCCcore/11.3.0 libtirpc/1.3.2 OpenMPI/4.1.4 Python/3.10.4 R/4.2.3
module unload SciPy-bundle/2022.05
export USE_NUMEXPR=0
export USE_BOTTLENECK=0

cd /net/people/plgrid/plgkpiechocki/TCLB

TCLB_EXEC="/net/people/plgrid/plgkpiechocki/TCLB/CLB/sw/main"
SCRIPT_DIR="kp/shallow_water/thesis/topology_validation/pipeline"
THESIS_DIR="kp/shallow_water/thesis"
OUT_BASE="output/sweep_scale_${SCALE}x"
SWEEP_RES_DIR="${THESIS_DIR}/wyniki_sweep_${SCALE}x"
XML_DIR="${SWEEP_RES_DIR}/xml"
mkdir -p $OUT_BASE $SWEEP_RES_DIR $XML_DIR

# Osobne pliki STL dla topologii pierwotnej i oczyszczonej (brak nadpisywania)
STL_ORIG_SCALED="${THESIS_DIR}/topology_orig_${SCALE}x.stl"
STL_CLEAN_SCALED="${THESIS_DIR}/topology_clean_${SCALE}x.stl"
python3 $SCRIPT_DIR/stl_rescale.py $SCALE "${THESIS_DIR}/basic_mesh_topology_00.stl" $STL_ORIG_SCALED
python3 $SCRIPT_DIR/stl_rescale.py $SCALE "${THESIS_DIR}/cleaned_topology_01.stl" $STL_CLEAN_SCALED

# Modele do sweepu (po odpowiedzi promotora mozesz zostawic wszystkie lub wybrany podzbior)
MODELS=("topology_orig" "topology_clean" "euve" "empty_cav")

for LAM in $(seq 200 20 400); do
    WAVE_CSV="${SWEEP_RES_DIR}/inlet_wave_lam${LAM}_${SCALE}x.csv"
    python3 $SCRIPT_DIR/gen_singlewave.py $SCALE $LAM $WAVE_CSV $ITERS

    for model_name in "${MODELS[@]}"; do
        if [ "$model_name" == "topology_orig" ]; then
            tpl_source="$SCRIPT_DIR/topology.xml.tpl"
            stl_arg="$STL_ORIG_SCALED"
        elif [ "$model_name" == "topology_clean" ]; then
            tpl_source="$SCRIPT_DIR/topology.xml.tpl"
            stl_arg="$STL_CLEAN_SCALED"
        elif [ "$model_name" == "euve" ]; then
            tpl_source="$SCRIPT_DIR/euve.xml.tpl"
            stl_arg="$STL_ORIG_SCALED"
        else
            tpl_source="$SCRIPT_DIR/empty_cavities.xml.tpl"
            stl_arg="$STL_ORIG_SCALED"
        fi

        run_name="${model_name}_lam${LAM}_${SCALE}x"
        run_out_dir="${OUT_BASE}/${run_name}/"
        mkdir -p $run_out_dir
        temp_xml="${XML_DIR}/${run_name}.xml"

        python3 $SCRIPT_DIR/prepare_xmls.py $SCALE $tpl_source $temp_xml $run_out_dir $WAVE_CSV $stl_arg $ITERS
        mpirun $TCLB_EXEC $temp_xml

        python3 $SCRIPT_DIR/diagnostics.py $run_out_dir "sweep_${model_name}_lam${LAM}" $SCALE $SWEEP_RES_DIR $LAM
        rm -f ${run_out_dir}*.vti ${run_out_dir}*.pvti
    done
done