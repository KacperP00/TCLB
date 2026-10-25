import sys
import os
import pyvista as pv

# Argumenty: 1: mnoznik skali, 2: wejsciowy plik STL (1.0x), 3: wyjsciowy plik STL
scale_factor = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
in_file = sys.argv[2] if len(sys.argv) > 2 else "kp/shallow_water/thesis/basic_mesh_topology_00.stl"
out_file = sys.argv[3] if len(sys.argv) > 3 else f"kp/shallow_water/thesis/topology_{scale_factor}x.stl"

os.makedirs(os.path.dirname(out_file), exist_ok=True)
mesh = pv.read(in_file)

# Skalowanie wspolrzednych X i Y wzgledem (0,0), os Z pozostaje bez zmian
mesh.points[:, 0] *= scale_factor
mesh.points[:, 1] *= scale_factor

mesh.save(out_file)
print(f"Zapisano przeskalowany STL ({scale_factor}x): {out_file}, zakres: {mesh.bounds}")