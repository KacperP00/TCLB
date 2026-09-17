import pyvista as pv
import glob

# Wyszukiwanie wynikowych plikow VTK
files = sorted(glob.glob('results/*.vtk'))

for f in files:
    # Wczytanie danych siatki
    mesh = pv.read(f)
    
    # Izolacja fazy rozproszonej na podstawie progu gestosci
    # W modelu d2q9_mcmp polem fazy drugiej jest standardowo 'rho2'
    drop = mesh.threshold(0.5, scalars='rho2')
    
    # Weryfikacja obecnosci fazy w domenie
    if drop.n_cells > 0:
        # Obliczenie pola powierzchni w 2D
        area = drop.volume
        
        # Wyznaczenie srodka masy kropli
        com = drop.center_of_mass()
        print(f"{f}: Pole = {area:.2f}, Srodek masy = {com}")
    else:
        print(f"{f}: Brak zdefiniowanej fazy")