#!/bin/bash
set -euo pipefail

SOLVER="CLB/d2q9_csf_viscstep/main"
DIAG_DIR="projects/aerospace_injector/benchmarks/diag"
mkdir -p "$DIAG_DIR"

# Test A: R=40 przy niskim nu (0.025/0.012), sledzenie relaksacji do 30k it
A_DIR="${DIAG_DIR}/laplace_R40_long"
mkdir -p "${A_DIR}/output" && rm -f "${A_DIR}/output"/*
cat > "${A_DIR}/config.xml" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${A_DIR}/output/" permissive="true">
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
        <Param name="Mobility" value="0.010"/>
        <Param name="IntWidth" value="0.333"/>
        <Param name="SurfaceTensionRate" value="0.0"/>
        <Param name="PF_Advection_Switch" value="0"/>
        <Param name="Pressure" value="0.0"/>
    </Model>
    <Solve Iterations="500"/>
    <Param name="SurfaceTensionRate" value="0.090"/>
    <Param name="PF_Advection_Switch" value="1"/>
    <Solve Iterations="30000">
        <VTK Iterations="3000"/>
        <Failcheck Iterations="3000"/>
    </Solve>
</CLBConfig>
EOF
echo "[Test A] Laplace R=40, nu=0.025/0.012, 30000 it..."
"$SOLVER" "${A_DIR}/config.xml" > "${A_DIR}/solver.log" 2>&1

# Test B: R=40 przy wysokim nu (0.1667, tau_ac=3890 it), 15k it
B_DIR="${DIAG_DIR}/laplace_R40_highnu"
mkdir -p "${B_DIR}/output" && rm -f "${B_DIR}/output"/*
cat > "${B_DIR}/config.xml" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${B_DIR}/output/" permissive="true">
    <Geometry nx="160" ny="160" predef="none" model="MRT">
        <MRT><Box/></MRT>
        <None name="drop"><OffgridSphere x="80.0" y="80.0" z="0.5" R="40.0"/></None>
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
        <VTK Iterations="3000"/>
        <Failcheck Iterations="3000"/>
    </Solve>
</CLBConfig>
EOF
echo "[Test B] Laplace R=40, nu=0.1667, 15000 it..."
"$SOLVER" "${B_DIR}/config.xml" > "${B_DIR}/solver.log" 2>&1

# Test C: Poiseuille 320x40 z inicjalizacja predkosci w calym kanale (brak impulsu)
C_DIR="${DIAG_DIR}/poiseuille_init"
mkdir -p "${C_DIR}/output" && rm -f "${C_DIR}/output"/*
cat > "${C_DIR}/config.xml" <<EOF
<?xml version="1.0"?>
<CLBConfig version="2.0" output="${C_DIR}/output/" permissive="true">
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
    <Solve Iterations="18000">
        <VTK Iterations="18000"/>
        <Failcheck Iterations="3000"/>
    </Solve>
</CLBConfig>
EOF
echo "[Test C] Poiseuille 320x40 z inicjalizacja VelocityX=0.028..."
"$SOLVER" "${C_DIR}/config.xml" > "${C_DIR}/solver.log" 2>&1

# Analiza wynikow wszystkich trzech testow
python3 - << 'PYEOF'
import glob, os, re
import numpy as np
import pyvista as pv

DIAG_DIR = "projects/aerospace_injector/benchmarks/diag"

def inspect_laplace_series(title, subdir):
    print(f"\n=== {title} (teoria: dP = 0.002250, sigma = 0.0900) ===")
    print(f"{'iter':>6s} | {'R_eq':>6s} | {'dP_mean':>9s} | {'dP_cc':>9s} | {'sigma_eff':>9s} | {'std_in':>8s} | {'max|U_sp|':>9s}")
    print("-" * 74)
    files = sorted(glob.glob(os.path.join(DIAG_DIR, subdir, "output", "*_VTK_P00_*.vti")))
    for f in files:
        it = int(re.search(r"_(\d+)\.vti$", f).group(1))
        m = pv.read(f)
        ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
        pf = m["PhaseField"].reshape((ny, nx))
        rho = m["Rho"].reshape((ny, nx))
        u = m["U"].reshape((ny, nx, 3))
        r_eq = np.sqrt(np.sum(pf > 0.0) / np.pi)
        yy, xx = np.mgrid[0:ny, 0:nx]
        rg = np.hypot(xx - (nx / 2.0 - 0.5), yy - (ny / 2.0 - 0.5))
        rin = rho[rg < r_eq - 7.0]
        rout = rho[rg > r_eq + 12.0]
        dp_mean = (float(np.mean(rin)) - float(np.mean(rout))) / 3.0
        dp_cc = (float(rho[ny // 2, nx // 2]) - float(rho[0, 0])) / 3.0
        u_sp = float(np.max(np.linalg.norm(u, axis=2)))
        print(f"{it:6d} | {r_eq:6.2f} | {dp_mean:9.6f} | {dp_cc:9.6f} | {dp_mean*r_eq:9.5f} | {np.std(rin):8.1e} | {u_sp:9.6f}")

inspect_laplace_series("TEST A: Laplace R=40, nu=0.025/0.012", "laplace_R40_long")
inspect_laplace_series("TEST B: Laplace R=40, nu=0.1667", "laplace_R40_highnu")

print("\n=== TEST C: Poiseuille 320x40 (inicjalizacja VelocityX=0.028) ===")
vti_c = sorted(glob.glob(os.path.join(DIAG_DIR, "poiseuille_init", "output", "*_VTK_P00_*.vti")))[-1]
m = pv.read(vti_c)
ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
u = m["U"].reshape((ny, nx, 3))
rho = m["Rho"].reshape((ny, nx))
for xs in [1, 100, 250, 318]:
    ux_col = u[1:-1, xs, 0]
    rho_col = rho[1:-1, xs]
    print(f"  x={xs:3d}: U_sr={ux_col.mean():.6f} | rho_sr={rho_col.mean():.5f} | Q_mas={np.sum(rho_col * ux_col):.5f}")
ux_sim = u[1:-1, 250, 0]
H = float(ny - 2)
eta = (np.arange(1, ny - 1, dtype=float) - 0.5) / H
ux_ana_tgt = 6.0 * 0.028 * eta * (1.0 - eta)
ux_ana_loc = 6.0 * ux_sim.mean() * eta * (1.0 - eta)
l2_abs = 100.0 * np.linalg.norm(ux_sim - ux_ana_tgt) / np.linalg.norm(ux_ana_tgt)
l2_shp = 100.0 * np.linalg.norm(ux_sim - ux_ana_loc) / np.linalg.norm(ux_ana_loc)
print(f"  x=250: U_sr={ux_sim.mean():.6f} ({(ux_sim.mean()-0.028)/0.028*100:+.2f}%) | L2 ksztalt={l2_shp:.3f}% | L2 calkowity={l2_abs:.3f}%")
PYEOF