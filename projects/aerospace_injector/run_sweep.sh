#!/bin/bash
set -euo pipefail

SOLVER="CLB/d2q9_csf_viscstep/main"
BASE_DIR="projects/aerospace_injector/sweep"
ANALYZER="projects/aerospace_injector/analyze_droplets.py"

# Parametric sweep cases: continuous-phase inlet velocity and safe iteration count
CASES=(
  "0.018 54000"
  "0.023 42000"
  "0.028 36000"
  "0.035 30000"
  "0.042 26000"
)

mkdir -p "$BASE_DIR"

for entry in "${CASES[@]}"; do
  read -r UC ITERS <<< "$entry"
  CASE_DIR="${BASE_DIR}/Uc_${UC}"
  OUT_DIR="${CASE_DIR}/output"
  CFG="${CASE_DIR}/config.xml"

  mkdir -p "$OUT_DIR"
  rm -f "${OUT_DIR}"/*

  cat > "$CFG" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${OUT_DIR}/" permissive="true">
    <Geometry nx="1280" ny="64" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <WVelocity name="inlet_main"><Box nx="1" dy="25" fy="-2"/></WVelocity>
        <EPressure name="outlet"><Box dx="-1" nx="1" dy="25" fy="-2"/></EPressure>
        <SVelocity name="inlet_jet"><Box dx="60" nx="32" dy="0" ny="1"/></SVelocity>
        <None name="fuel_nozzle"><Box dx="60" nx="32" dy="1" ny="24"/></None>
        <Wall name="walls" mask="ALL">
            <Box dy="-1" ny="1"/>
            <Box dx="0" nx="60" dy="0" ny="25"/>
            <Box dx="92" dy="0" ny="25"/>
        </Wall>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="PhaseField" value="0.5" zone="inlet_jet"/>
        <Param name="PhaseField" value="0.5" zone="fuel_nozzle"/>
        <Param name="Viscosity" value="0.025"/>
        <Param name="Viscosity_l" value="0.012"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
        <Param name="nubuffer" value="0.025"/>
        <Param name="Mobility" value="0.010"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.090"/>
        <Param name="WettingAngle" value="2.50"/>
        <Param name="WallAdhesionDecay" value="0.7"/>
        <Param name="PF_Advection_Switch" value="1"/>
        <Param name="VelocityX" value="0.0"/>
        <Param name="VelocityY" value="0.0"/>
        <Param name="VelocityX" value="${UC}" zone="inlet_main"/>
        <Param name="VelocityY" value="0.008" zone="inlet_jet"/>
        <Param name="VelocityY" value="0.008" zone="fuel_nozzle"/>
        <Param name="Pressure" value="0.0" zone="outlet"/>
    </Model>
    <Solve Iterations="${ITERS}">
        <VTK Iterations="500"/>
        <Failcheck Iterations="500"/>
    </Solve>
</CLBConfig>
EOF

  echo "============================================================"
  echo "Running case: U_c=${UC}, Iterations=${ITERS}"
  echo "============================================================"

  "$SOLVER" "$CFG" > "${CASE_DIR}/solver.log" 2>&1

  python3 "$ANALYZER" \
    --input-dir "$OUT_DIR" \
    --out-json "${CASE_DIR}/summary.json" \
    --plot-dir "$CASE_DIR" \
    --uc "$UC" \
    --auto-probes
done

python3 projects/aerospace_injector/plot_sweep.py