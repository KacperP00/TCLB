import numpy as np
import pandas as pd

# --- PARAMETRY WEJŚCIOWE ---
GRAVITY = 0.0005
HEIGHT = 40.0
AMPLITUDE = 0.002
WAVE_LENGTH = 1200 # Sugerowany okres dla fali płaskiej i rezonansu
NY = 352
L = 320 # Szerookość kanału (lu)
WALL_WIDTH = 16      # Szerokość ścian bocznych (dy)
CAVITY_NX = 1280      # Rozmiar wnęki (64cm)
NO_PERIODS_IN_CAVITYZONE = 5
k = 2.0 * np.pi / WAVE_LENGTH
WAVE_PHASE = 0.0
k_imag = 0.0

def generate_xml_geometry(nx_up, nx_cav, nx_down, ny, wall):
    total_nx = nx_up + nx_cav + nx_down
    
    def get_header(mode_name):
        return f"\n"

    def get_common_elements():
        return (
            f'        <MRT><Box/></MRT>\n'
            f'        <WPressure name="Inlet">\n'
            f'            <Box dx="0" nx="1" dy="{wall}" ny="{ny-2*wall}"/>\n'
            f'        </WPressure>\n'
            f'        <Wall name="Outlet">\n'
            f'            <Box dx="{total_nx-1}" nx="1"/>\n'
            f'        </Wall>'
        )

    # Wersja 1: Czysty kanał (ściany ciągłe)
    clean_xml = get_header("CZYSTY KANAŁ") + "\n    <Geometry nx=\"{}\" ny=\"{}\">\n".format(total_nx, ny)
    clean_xml += get_common_elements() + "\n"
    clean_xml += (
        f'        <Wall>\n'
        f'            <Box dy="0" ny="{wall}"/>\n'
        f'            <Box dy="{ny-wall}" ny="{wall}"/>\n'
        f'        </Wall>\n'
        f'        <Obj2 name="Reflection_Zone">\n'
        f'            <Box dx="{nx_up//2+nx_up//5}" nx="{nx_up//5}" dy="{ny//2 - 16}" ny="32"/>\n'
        f'        </Obj2>\n'
        f'    </Geometry>'
    )


    """
    # Wersja 2: Kanał z wnękami
    cavity_xml = get_header("KANAŁ Z WNĘKAMI") + "\n    <Geometry nx=\"{}\" ny=\"{}\">\n".format(total_nx, ny)
    cavity_xml += get_common_elements() + "\n"
    cavity_xml += (
        f'        \n'
        f'        <Wall>\n'
        f'            <Box dx="0" nx="{nx_up}" dy="0" ny="{wall}"/>\n'
        f'            <Box dx="0" nx="{nx_up}" dy="{ny-wall}" ny="{wall}"/>\n'
        f'        </Wall>\n'
        f'        \n'
        f'        <Wall>\n'
        f'            <Box dx="{nx_up + nx_cav}" nx="{nx_down}" dy="0" ny="{wall}"/>\n'
        f'            <Box dx="{nx_up + nx_cav}" nx="{nx_down}" dy="{ny-wall}" ny="{wall}"/>\n'
        f'        </Wall>\n'
        f'        <Obj2 name="Reflection_Zone">\n'
        f'            <Box dx="{nx_up//2}" nx="320" dy="{ny//2 - 16}" ny="32"/>\n'
        f'        </Obj2>\n'
        f'        <Obj3 name="Transmission_Zone">\n'
        f'            <Box dx="{nx_up + nx_cav + 500}" nx="320" dy="{ny//2 - 16}" ny="32"/>\n'
        f'        </Obj3>\n'
        f'    </Geometry>'
    )
    """
    return clean_xml

# --- OBLICZENIA ---
c = np.sqrt(GRAVITY * HEIGHT)
period = WAVE_LENGTH / c


# Logika 6T i 5T
nx_upstream = int(np.round((NO_PERIODS_IN_CAVITYZONE+1) * WAVE_LENGTH))
nx_downstream = int(np.round(NO_PERIODS_IN_CAVITYZONE * WAVE_LENGTH))

# Generowanie plików CSV (istniejąca logika)
t = np.arange(12*period*1.2)
omega = 2.0 * np.pi / period
wave_h = AMPLITUDE * np.cos(omega * t)
df = pd.DataFrame({'iter': t, 'cos': wave_h})
df.to_csv('kp/shallow_water/thesis/inlet_wave.csv', index=False)

# Generowanie XML
clean_geo = generate_xml_geometry(nx_upstream, CAVITY_NX, nx_downstream, NY, WALL_WIDTH)

print(f"Obliczona prędkość fali (c): {c:.6f} lu/tu")
print(f"Długość fali (lambda): {WAVE_LENGTH:.6f} lu")
print(f"Liczba falowa (k): {k:.6} 1/lu")
print(f"Częstość omega: {omega:.6f} 1/tu")
print(f"Kanał dolotowy: {nx_upstream} lu | Kanał wylotowy: {nx_downstream} lu")
print(f"Okres fali: {period:.6f}")
print(f"Ilość iteracji: {12 * period:.6f}")
print(f"kL/pi: {k*L/np.pi:.6f} <? 1")
print(f"k*h: {k * HEIGHT:.6f} <? 0.3")
print(clean_geo)
#print(cav_geo)



print(
    f'      <Param name="Height" value="{HEIGHT:.6f}"/>\n'
    f'      <Param name="Wave_A" value="{AMPLITUDE:.6f}"/>\n'
    f'      <Param name="Wave_k_real" value="{k:.6f}"/>\n'
    f'      <Param name="Wave_k_imag" value="{k_imag:.6f}"/>\n'
    f'      <Param name="Wave_w" value="{omega:.6f}"/>\n'
    f'      <Param name="Wave_Phase" value="{WAVE_PHASE:.6f}"/>\n'
    f'      <Param name="Wave_Period" value="{period:.6f}"/>\n'
    f'      <Param name="Wave_Length" value="{WAVE_LENGTH:.6f}"/>\n')