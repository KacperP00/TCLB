import sys
import pyvista as pv

def clean_topology_stl(input_file, output_file, min_area=50.0):
    print(f"Wczytywanie siatki STL: {input_file}...")
    mesh = pv.read(input_file)
    
    bodies = mesh.split_bodies()
    print(f"Znaleziono {len(bodies)} niezaleznych fragmentow (wysp).")
    
    # Sortowanie wysp malejaco po polu powierzchni
    bodies_sorted = sorted(bodies, key=lambda b: b.area, reverse=True)
    
    print("\n--- TOP 5 najwiekszych elementow ---")
    for i, b in enumerate(bodies_sorted[:5]):
        print(f"Element {i+1}: Pole powierzchni (area) = {b.area:.1f}, Objetosc 3D = {b.volume:.1f}")

    # Filtracja szumu (zostawiamy elementy o polu > min_area)
    valid_bodies = []
    for body in bodies_sorted:
        if body.area >= min_area:
            valid_bodies.append(body)
            
    kept_count = len(valid_bodies)
    removed_count = len(bodies) - kept_count
    
    if kept_count == 0:
        print(f"\nBlad: Zaden z elementow nie spelnil warunku area >= {min_area}.")
        return
        
    print(f"\nLaczenie {kept_count} glownych elementow (odrzucono {removed_count} drobin szumu)...")
    
    # Zespolenie przefiltrowanych wysp
    print(f"\nLaczenie {kept_count} glownych elementow (odrzucono {removed_count} drobin szumu)...")
    
    # Zespolenie przefiltrowanych wysp
    cleaned_mesh = valid_bodies[0]
    for body in valid_bodies[1:]:
        cleaned_mesh = cleaned_mesh.merge(body)
        
    # Konwersja z powrotem na siatkę powierzchniową (PolyData) zgodną z STL
    cleaned_mesh = cleaned_mesh.extract_surface().triangulate()
        
    cleaned_mesh.save(output_file)
    print(f"Zapisano oczyszczona geometrie do: {output_file}")

if __name__ == "__main__":
    in_stl = sys.argv[1] if len(sys.argv) > 1 else "kp/shallow_water/thesis/basic_mesh_topology_00.stl"
    out_stl = sys.argv[2] if len(sys.argv) > 2 else "kp/shallow_water/thesis/cleaned_topology_01.stl"
    
    # Prog 50.0 to bezpieczny margines na pole powierzchni (area) dla mikro-drobin wokselowych
    clean_topology_stl(in_stl, out_stl, min_area=50.0)