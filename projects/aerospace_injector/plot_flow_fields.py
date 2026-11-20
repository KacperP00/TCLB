import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

CASE_DIR = "projects/aerospace_injector/sweep/Uc_0.028"
UC = 0.028

files = sorted(glob.glob(os.path.join(CASE_DIR, "output", "*_VTK_P00_*.vti")))
if not files:
  raise SystemExit(f"No VTI files found in {CASE_DIR}/output")

vti_path = files[-1]
m = pv.read(vti_path)
ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1

# Construct wall mask for 1280x64 geometry
wall_mask = np.zeros((ny, nx), dtype=bool)
wall_mask[-1, :] = True
wall_mask[:25, :60] = True
wall_mask[:25, 92:] = True

pf = m["PhaseField"].reshape((ny, nx))
u = m["U"].reshape((ny, nx, 3))
rho = m["Rho"].reshape((ny, nx))

ux = u[:, :, 0].copy()
uy = u[:, :, 1].copy()
umag_norm = np.hypot(ux, uy) / UC
dp = (rho - 1.0) / 3.0

# Zero velocity inside solid walls for clean streamline tracing
ux[wall_mask] = 0.0
uy[wall_mask] = 0.0

fig, axes = plt.subplots(3, 1, figsize=(13.5, 6.8), sharex=True)

# Panel 1: Phase field
cmap_pf = plt.cm.coolwarm.copy()
cmap_pf.set_bad(color="#2b2b2b")
pf_m = np.ma.masked_where(wall_mask, pf)
im0 = axes[0].imshow(pf_m, origin="lower", cmap=cmap_pf, vmin=-0.5, vmax=0.5)
axes[0].contour(pf_m, levels=[0.0], colors="black", linewidths=0.8)
axes[0].set_title("Phase Field $\\phi$ and Zero-Contour Interface ($U_c = 0.028$, $Ca = 0.0078$, $t = 36000$)")
axes[0].set_ylabel("y [LU]")
plt.colorbar(im0, ax=axes[0], label="$\\phi$ [-]", pad=0.015)

# Panel 2: Normalized velocity magnitude with streamlines
cmap_u = plt.cm.viridis.copy()
cmap_u.set_bad(color="#2b2b2b")
u_m = np.ma.masked_where(wall_mask, umag_norm)
im1 = axes[1].imshow(u_m, origin="lower", cmap=cmap_u, vmin=0.0, vmax=2.0)
axes[1].contour(pf_m, levels=[0.0], colors="white", linewidths=0.7, alpha=0.85)
axes[1].streamplot(
    np.arange(nx, dtype=float), np.arange(ny, dtype=float), ux, uy,
    color="white", density=(2.2, 0.55), linewidth=0.55, arrowsize=0.6
)
axes[1].set_title("Normalized Velocity Magnitude $|\\mathbf{U}| / U_c$ with Streamlines")
axes[1].set_ylabel("y [LU]")
plt.colorbar(im1, ax=axes[1], label="$|\\mathbf{U}| / U_c$ [-]", pad=0.015)

# Panel 3: Local pressure perturbation (axial mean removed to expose Laplace jump)
cmap_p = plt.cm.RdBu_r.copy()
cmap_p.set_bad(color="#2b2b2b")
dp_col_mean = np.array([
    np.mean(dp[~wall_mask[:, j], j]) if np.any(~wall_mask[:, j]) else 0.0
    for j in range(nx)
])
dp_local = dp - dp_col_mean[None, :]
dp_m = np.ma.masked_where(wall_mask, dp_local)
p_lim = float(np.percentile(np.abs(dp_local[~wall_mask]), 99.0))
im2 = axes[2].imshow(dp_m, origin="lower", cmap=cmap_p, vmin=-p_lim, vmax=p_lim)
axes[2].contour(pf_m, levels=[0.0], colors="black", linewidths=0.7)
axes[2].set_title("Local Capillary Pressure Perturbation $\\Delta P_{\\mathrm{loc}} = \\Delta P - \\langle \\Delta P \\rangle_y(x)$")
axes[2].set_ylabel("y [LU]")
axes[2].set_xlabel("Axial Position $x$ [LU]")
plt.colorbar(im2, ax=axes[2], label="$\\Delta P_{\\mathrm{loc}}$ [LU]", pad=0.015)

plt.tight_layout()
out_path = os.path.join(CASE_DIR, "flow_fields_baseline.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved multi-field baseline visualization: {out_path}")