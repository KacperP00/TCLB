#!/bin/bash
set -euo pipefail

SOLVER_CUM="CLB/d2q9_csf_viscstep_cumulant/main"
CMP_DIR="projects/aerospace_injector/cumulant_cmp"
BM_CUM_DIR="projects/aerospace_injector/benchmarks_cumulant"

if [ ! -x "$SOLVER_CUM" ]; then
  echo "Error: ${SOLVER_CUM} not found. Compile first with: make d2q9_csf_viscstep_cumulant"
  exit 1
fi

mkdir -p "$CMP_DIR"

# 1. Run standard 3-benchmark suite with Cumulant operator
bash projects/aerospace_injector/run_benchmarks.sh "$SOLVER_CUM" "$BM_CUM_DIR" "Cumulant"

# 2. Low-viscosity Young-Laplace check (R=40, nu=0.025/0.012) to verify bulk-viscosity damping
LOW_DIR="${CMP_DIR}/laplace_R40_lownu"
mkdir -p "${LOW_DIR}/output" && rm -f "${LOW_DIR}/output"/*
cat > "${LOW_DIR}/config.xml" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${LOW_DIR}/output/" permissive="true">
    <Geometry nx="160" ny="160" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <None name="drop"><OffgridSphere x="80.0" y="80.0" z="0.5" R="40.0"/></None>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="PhaseField" value="0.5" zone="drop"/>
        <Param name="Viscosity" value="0.025"/>
        <Param name="Viscosity_l" value="0.012"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
        <Param name="nubuffer" value="0.025"/>
        <Param name="Mobility" value="0.010"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.0"/>
        <Param name="PF_Advection_Switch" value="0"/>
        <Param name="Pressure" value="0.0"/>
    </Model>
    <Solve Iterations="500"/>
    <Param name="SurfaceTensionRate" value="0.090"/>
    <Param name="PF_Advection_Switch" value="1"/>
    <Solve Iterations="15000">
        <VTK Iterations="15000"/>
        <Failcheck Iterations="3000"/>
    </Solve>
</CLBConfig>
EOF
echo "[Cumulant] Running low-viscosity Laplace R=40 check (nu=0.025/0.012, 15000 it)..."
"$SOLVER_CUM" "${LOW_DIR}/config.xml" > "${LOW_DIR}/solver.log" 2>&1

# 3. Baseline T-junction micro-injector (1280x64, U_c=0.028, 36000 it)
INJ_DIR="${CMP_DIR}/Uc_0.028"
OUT_DIR="${INJ_DIR}/output"
CFG="${INJ_DIR}/config.xml"
mkdir -p "$OUT_DIR" && rm -f "${OUT_DIR}"/*

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
        <Param name="VelocityX" value="0.028" zone="inlet_main"/>
        <Param name="VelocityY" value="0.008" zone="inlet_jet"/>
        <Param name="VelocityY" value="0.008" zone="fuel_nozzle"/>
        <Param name="Pressure" value="0.0" zone="outlet"/>
    </Model>
    <Solve Iterations="36000">
        <VTK Iterations="500"/>
        <Failcheck Iterations="500"/>
    </Solve>
</CLBConfig>
EOF

echo "[Cumulant] Running baseline T-junction (1280x64, U_c=0.028, 36000 it)..."
"$SOLVER_CUM" "$CFG" > "${INJ_DIR}/solver.log" 2>&1

python3 projects/aerospace_injector/analyze_droplets.py \
  --input-dir "$OUT_DIR" \
  --out-json "${INJ_DIR}/summary.json" \
  --plot-dir "$INJ_DIR" \
  --uc 0.028 \
  --auto-probes

python3 projects/aerospace_injector/analyze_cumulant.py