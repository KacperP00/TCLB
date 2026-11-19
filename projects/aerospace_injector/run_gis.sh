#!/bin/bash
set -euo pipefail

SOLVER="CLB/d2q9_csf_viscstep/main"
GIS_DIR="projects/aerospace_injector/gis_fine"
OUT_DIR="${GIS_DIR}/output"
CFG="${GIS_DIR}/config_fine.xml"

mkdir -p "$OUT_DIR"
rm -f "${OUT_DIR}"/*

# Generacja konfiguracji siatki 1920x96 (r=1.5, skalowanie przy stalym ST=0.090)
cat > "$CFG" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${OUT_DIR}/" permissive="true">
    <Geometry nx="1920" ny="96" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <WVelocity name="inlet_main"><Box nx="1" dy="38" fy="-2"/></WVelocity>
        <EPressure name="outlet"><Box dx="-1" nx="1" dy="38" fy="-2"/></EPressure>
        <SVelocity name="inlet_jet"><Box dx="90" nx="48" dy="0" ny="1"/></SVelocity>
        <None name="fuel_nozzle"><Box dx="90" nx="48" dy="1" ny="37"/></None>
        <Wall name="walls" mask="ALL">
            <Box dy="-1" ny="1"/>
            <Box dx="0" nx="90" dy="0" ny="38"/>
            <Box dx="138" dy="0" ny="38"/>
        </Wall>
    </Geometry>
    <Model>
        <Param name="PhaseField" value="-0.5"/>
        <Param name="PhaseField" value="0.5" zone="inlet_jet"/>
        <Param name="PhaseField" value="0.5" zone="fuel_nozzle"/>
        <Param name="Viscosity" value="0.030619"/>
        <Param name="Viscosity_l" value="0.014697"/>
        <Param name="ViscosityStepWidth" value="0.25"/>
        <Param name="Mobility" value="0.012247"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.090"/>
        <Param name="WettingAngle" value="2.50"/>
        <Param name="WallAdhesionDecay" value="0.7"/>
        <Param name="PF_Advection_Switch" value="1"/>
        <Param name="VelocityX" value="0.0"/>
        <Param name="VelocityY" value="0.0"/>
        <Param name="VelocityX" value="0.022862" zone="inlet_main"/>
        <Param name="VelocityY" value="0.006532" zone="inlet_jet"/>
        <Param name="VelocityY" value="0.006532" zone="fuel_nozzle"/>
        <Param name="Pressure" value="0.0" zone="outlet"/>
    </Model>
    <Solve Iterations="66000">
        <VTK Iterations="1000"/>
        <Failcheck Iterations="1000"/>
    </Solve>
</CLBConfig>
EOF

echo "Uruchamianie GIS (1920x96, 66000 iteracji, ok. 9-10 min)..."
"$SOLVER" "$CFG" > "${GIS_DIR}/solver.log" 2>&1

python3 projects/aerospace_injector/analyze_gis.py