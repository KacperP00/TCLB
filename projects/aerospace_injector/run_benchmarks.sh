#!/bin/bash
set -euo pipefail

SOLVER="CLB/d2q9_csf_viscstep/main"
BM_DIR="projects/aerospace_injector/benchmarks"

mkdir -p "$BM_DIR"

# 1. Benchmark Younga-Laplace'a (160x160, nu=1/6 dla wytlumienia fal akustycznych)
for R in 15 20 30 40; do
  CASE_DIR="${BM_DIR}/laplace_R${R}"
  OUT_DIR="${CASE_DIR}/output"
  CFG="${CASE_DIR}/config.xml"
  mkdir -p "$OUT_DIR"
  rm -f "${OUT_DIR}"/*

  cat > "$CFG" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${OUT_DIR}/" permissive="true">
    <Geometry nx="160" ny="160" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <None name="drop">
            <OffgridSphere x="80.0" y="80.0" z="0.5" R="${R}.0"/>
        </None>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="PhaseField" value="0.5" zone="drop"/>
        <Param name="Viscosity" value="0.166667"/>
        <Param name="Viscosity_l" value="0.166667"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
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

  echo "[Benchmark 1/3] Young-Laplace R=${R} LU (nu=1/6, 15000 it)..."
  "$SOLVER" "$CFG" > "${CASE_DIR}/solver.log" 2>&1
done

# 2. Benchmark kata zwilzania (Sessile Drop: 60, 90, 143.2 deg, WallAdhesionDecay=0.7)
for pair in "60 1.0472" "90 1.5708" "143 2.5000"; do
  read -r DEG RAD <<< "$pair"
  CASE_DIR="${BM_DIR}/sessile_${DEG}deg"
  OUT_DIR="${CASE_DIR}/output"
  CFG="${CASE_DIR}/config.xml"
  mkdir -p "$OUT_DIR"
  rm -f "${OUT_DIR}"/*

  cat > "$CFG" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${OUT_DIR}/" permissive="true">
    <Geometry nx="192" ny="96" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <None name="drop">
            <OffgridSphere x="96.0" y="1.0" z="0.5" R="30.0"/>
        </None>
        <Wall name="walls" mask="ALL">
            <Box dy="0" ny="1"/>
            <Box dy="-1" ny="1"/>
        </Wall>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="PhaseField" value="0.5" zone="drop"/>
        <Param name="Viscosity" value="0.025"/>
        <Param name="Viscosity_l" value="0.012"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
        <Param name="Mobility" value="0.010"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.0"/>
        <Param name="WettingAngle" value="${RAD}"/>
        <Param name="WallAdhesionDecay" value="0.7"/>
        <Param name="PF_Advection_Switch" value="0"/>
        <Param name="Pressure" value="0.0"/>
    </Model>
    <Solve Iterations="500"/>
    <Param name="SurfaceTensionRate" value="0.090"/>
    <Param name="PF_Advection_Switch" value="1"/>
    <Solve Iterations="12000">
        <VTK Iterations="12000"/>
        <Failcheck Iterations="1000"/>
    </Solve>
</CLBConfig>
EOF

  echo "[Benchmark 2/3] Sessile Drop theta=${DEG} deg..."
  "$SOLVER" "$CFG" > "${CASE_DIR}/solver.log" 2>&1
done

# 3. Benchmark profilu Poiseuille'a (320x40, H=38, z inicjalizacja predkosci)
CASE_DIR="${BM_DIR}/poiseuille"
OUT_DIR="${CASE_DIR}/output"
CFG="${CASE_DIR}/config.xml"
mkdir -p "$OUT_DIR"
rm -f "${OUT_DIR}"/*

cat > "$CFG" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${OUT_DIR}/" permissive="true">
    <Geometry nx="320" ny="40" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <None name="fluid_init"><Box dy="1" fy="-2"/></None>
        <WVelocity name="inlet"><Box nx="1" dy="1" fy="-2"/></WVelocity>
        <EPressure name="outlet"><Box dx="-1" nx="1" dy="1" fy="-2"/></EPressure>
        <Wall name="walls" mask="ALL">
            <Box dy="0" ny="1"/>
            <Box dy="-1" ny="1"/>
        </Wall>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="Viscosity" value="0.025"/>
        <Param name="Viscosity_l" value="0.012"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
        <Param name="Mobility" value="0.010"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.0"/>
        <Param name="PF_Advection_Switch" value="0"/>
        <Param name="VelocityX" value="0.0"/>
        <Param name="VelocityX" value="0.028" zone="fluid_init"/>
        <Param name="VelocityX" value="0.028" zone="inlet"/>
        <Param name="Pressure" value="0.0" zone="outlet"/>
    </Model>
    <Solve Iterations="25000">
        <VTK Iterations="25000"/>
        <Failcheck Iterations="5000"/>
    </Solve>
</CLBConfig>
EOF

echo "[Benchmark 3/3] 2D Poiseuille Channel (320x40, 25000 it)..."
"$SOLVER" "$CFG" > "${CASE_DIR}/solver.log" 2>&1

python3 projects/aerospace_injector/analyze_benchmarks.py