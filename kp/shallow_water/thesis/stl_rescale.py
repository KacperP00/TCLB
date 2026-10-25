import sys
import pyvista as pv

def scale_stl_2d(input_stl, output_stl, scale_factor=1.5):
    # Wczytanie siatki trojkatow z pliku STL
    mesh = pv.read(input_stl)
    
    print(f"Zakres przed skalowaniem (X, Y, Z): {mesh.bounds}")
    
    # Skalowanie wspolrzednych X i Y wzgledem (0,0), os Z pozostaje bez zmian (1.0)
    mesh.points[:, 0] *= scale_factor
    mesh.points[:, 1] *= scale_factor
    
    print(f"Zakres po skalowaniu (X, Y, Z):    {mesh.bounds}")
    
    # Zapis przeskalowanej geometrii do nowego pliku STL
    mesh.save(output_stl)
    print(f"Zapisano do: {output_stl}")

if __name__ == "__main__":
    in_file = "kp/shallow_water/thesis/basic_mesh_topology_00.stl"
    out_file = "kp/shallow_water/thesis/dense_mesh_topology_00.stl"
    
    # Mozliwosc nadpisania mnoznika argumentem z terminala (domyslnie 1.5)
    factor = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    scale_stl_2d(in_file, out_file, factor)