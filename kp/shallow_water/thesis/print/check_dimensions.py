import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, cKDTree

# Parametry skali i wysokosci roboczej wysp [mm]
L_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 160.0
H_ISLAND_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
SCALE_XY = L_MM / 100.0

INPUT_FILES = [
    ("Pierwotna", "kp/shallow_water/thesis/print/basic_mesh_topology_00.stl"),
    ("Oczyszczona", "kp/shallow_water/thesis/print/cleaned_topology_01.stl"),
]


def min_bbox_dims(pts_2d):
    # Minimalna szerokosc i dlugosc wyspy z krawedzi otoczki wypuklej
    if len(pts_2d) < 3:
        d = pts_2d.max(axis=0) - pts_2d.min(axis=0)
        return float(d.min()), float(d.max())
    hull = ConvexHull(pts_2d)
    h_pts = pts_2d[hull.vertices]
    best_min, best_pair = np.inf, (0.0, 0.0)
    for i in range(len(h_pts)):
        edge = h_pts[(i + 1) % len(h_pts)] - h_pts[i]
        norm = np.linalg.norm(edge)
        if norm == 0:
            continue
        u = edge / norm
        v = np.array([-u[1], u[0]])
        w1 = np.ptp(h_pts @ u)
        w2 = np.ptp(h_pts @ v)
        mn, mx = min(w1, w2), max(w1, w2)
        if mn < best_min:
            best_min, best_pair = mn, (float(mn), float(mx))
    return best_pair


def sample_slice_edges(slc_mesh, step_mm=0.15):
    # Probkowanie krawedzi przekroju 2D dla dokladnego pomiaru szczelin
    pts = slc_mesh.points[:, :2]
    lines = slc_mesh.lines
    samples = []
    idx = 0
    while idx < len(lines):
        n = lines[idx]
        ids = lines[idx + 1 : idx + 1 + n]
        idx += n + 1
        for k in range(len(ids) - 1):
            p0, p1 = pts[ids[k]], pts[ids[k + 1]]
            dist = np.linalg.norm(p1 - p0)
            m = max(1, int(np.ceil(dist / step_mm)))
            t = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
            samples.append(p0 + t * (p1 - p0))
    return np.vstack(samples) if samples else pts


for label, path in INPUT_FILES:
    if not os.path.exists(path):
        print(f"Pominieto (brak pliku): {path}")
        continue

    mesh = pv.read(path).extract_surface().triangulate()
    mesh.points[:, 0] *= SCALE_XY
    mesh.points[:, 1] *= SCALE_XY
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    z_height = max(zmax - zmin, 1e-9)
    z_mid = 0.5 * (zmin + zmax)

    # Podzial na osobne wyspy i wyznaczenie ich wymiarow
    conn = mesh.connectivity()
    region_ids = np.unique(conn["RegionId"])
    islands = []
    boundary_clouds = []

    for rid in region_ids:
        part = conn.threshold([rid, rid], scalars="RegionId").extract_surface()
        pts_xy = np.unique(np.round(part.points[:, :2], 4), axis=0)
        w_min, w_max = min_bbox_dims(pts_xy)
        area_mm2 = abs(part.volume) / z_height
        islands.append((w_min, w_max, area_mm2, int(rid)))

        slc = part.slice(normal=(0, 0, 1), origin=(0, 0, z_mid))
        if slc.n_points > 0:
            boundary_clouds.append(sample_slice_edges(slc))

    # Wyznaczenie najwezszej szczeliny miedzy roznymi wyspami
    min_gap = np.inf
    for i in range(len(boundary_clouds)):
        tree = cKDTree(boundary_clouds[i])
        for j in range(i + 1, len(boundary_clouds)):
            d, _ = tree.query(boundary_clouds[j])
            if d.min() < min_gap:
                min_gap = float(d.min())

    islands.sort(key=lambda x: x[0])
    print(f"\n=== {label} ({len(islands)} wysp, L = {L_MM:.0f} mm) ===")
    print(f"Najmniejsza szerokosc wyspy: {islands[0][0]:.2f} mm (smuklosc dla H={H_ISLAND_MM:.0f} mm: {H_ISLAND_MM / max(islands[0][0], 1e-6):.1f}:1)")
    print(f"Najmniejsze pole wyspy:      {min(x[2] for x in islands):.2f} mm^2")
    print(f"Najwezsza szczelina (gap):   {min_gap:.2f} mm")
    print("5 najcienszych wysp (min_szer x dlugosc [mm] | pole [mm^2]):")
    for w_min, w_max, area, rid in islands[:5]:
        print(f"  ID {rid:3d}: {w_min:5.2f} x {w_max:6.2f} mm | pole: {area:6.2f} mm^2")