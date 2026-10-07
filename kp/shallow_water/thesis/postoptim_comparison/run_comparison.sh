#!/bin/bash
# Zapisz jako run_cluster.sh i nadaj uprawnienia: chmod +x run_cluster.sh

TCLB_EXEC="./tclb"            
CORES=16                      
OUT_BASE="output/comparison"  
SCRIPT_DIR="kp/shallow_water/thesis" 

mkdir -p $OUT_BASE

echo "=== GENEROWANIE WYMUSZEŃ ==="
python3 $SCRIPT_DIR/singlewave_generator.py 294.0
python3 $SCRIPT_DIR/gauss_fin_4.py

declare -A models=(
    ["clean_canal"]="$SCRIPT_DIR/clean_canal.xml"
    ["euve"]="$SCRIPT_DIR/euve_comparison_2.xml"
    ["topology"]="$SCRIPT_DIR/topology_basic.xml"
    ["empty_cav"]="$SCRIPT_DIR/empty_cavities.xml"
)

# --- 1. SYMULACJE I ANALIZA PACZEK FALOWYCH ---
echo "=== ETAP 1: PACZKI FALOWE (SZEROKOPASMOWE) ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_packet"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> Uruchamianie LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    # Wstrzyknięcie folderu wyjściowego i pliku wymuszenia paczką
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    sed -i 's|<CSV file="[^"]*"|<CSV file="'"$SCRIPT_DIR/inlet_packet.csv"'"|g' $temp_xml
    
    mpirun -n $CORES $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT dla paczki falowej..."
    # Wywołanie: skrypt [ścieżka_do_katalogu_z_VTK] [nazwa_przypadku]
    python3 $SCRIPT_DIR/diagnostics_03.py $run_out_dir "packet_${model_name}"

    echo "-> Czyszczenie plików VTK dla $run_name..."
    rm ${run_out_dir}/*.vti
done

# --- 2. SYMULACJE I ANALIZA FALI CIĄGŁEJ (294 LU) ---
echo "=== ETAP 2: FALA CIĄGŁA (REZYDUALNA - 294 LU) ==="
for model_name in "${!models[@]}"; do
    xml_source="${models[$model_name]}"
    run_name="${model_name}_single"
    run_out_dir="${OUT_BASE}/${run_name}/"
    
    echo "-> Uruchamianie LBM: $run_name"
    mkdir -p $run_out_dir
    temp_xml="temp_${run_name}.xml"
    cp $xml_source $temp_xml
    
    # Wstrzyknięcie folderu wyjściowego i pliku wymuszenia fali ciągłej
    sed -i 's|output="[^"]*"|output="'"$run_out_dir"'"|g' $temp_xml
    sed -i 's|<CSV file="[^"]*"|<CSV file="'"$SCRIPT_DIR/inlet_wave.csv"'"|g' $temp_xml
    
    mpirun -n $CORES $TCLB_EXEC $temp_xml
    rm $temp_xml

    echo "-> Analiza 2D FFT dla fali ciągłej..."
    python3 $SCRIPT_DIR/singlewave_diagnostics.py $run_out_dir "single_${model_name}"

    echo "-> Czyszczenie plików VTK dla $run_name..."
    rm ${run_out_dir}/*.vti
done

echo "=== OBLICZENIA ZAKOŃCZONE POMYŚLNIE ==="