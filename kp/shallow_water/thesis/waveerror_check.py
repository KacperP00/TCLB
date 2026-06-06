import sys
import pandas as pd

# Walidacja argumentów wejściowych
if len(sys.argv) < 2:
    print("Użycie: python3 generuj_xml.py <lambda>")
    sys.exit(1)

target_lambda = float(sys.argv[1])

# Otwarcie pliku z danymi wejściowymi
try:
    df = pd.read_csv('kp/shallow_water/thesis/baza_fal.csv')
except FileNotFoundError:
    print("Brak pliku baza_fal.csv")
    sys.exit(1)

# Filtracja danych dla szukanej długości fali
row = df[df['lambda'] == target_lambda]

if row.empty:
    print(f"Brak danych dla lambda = {target_lambda}")
    sys.exit(1)

# Pobranie parametrów głównej fali (forward)
wave_a = row['A_forward'].values[0]
wave_k_real = row['k_real'].values[0]
wave_k_imag = row['k_imag'].values[0]
wave_w = row['omega'].values[0]
wave_phase = row['phase_forward'].values[0]
wave_period = row['period_iters'].values[0]

# Generowanie struktury XML z wymuszeniem zapisu dziesiętnego (litera 'f' w formacie)
xml_snippet = f"""          <Param name="Wave_A" value="{wave_a:.6f}"/>
        <Param name="Wave_k_real" value="{wave_k_real:.6f}"/>
        <Param name="Wave_k_imag" value="{wave_k_imag:.8f}"/>
        <Param name="Wave_w" value="{wave_w:.6f}"/>
        <Param name="Wave_Phase" value="{wave_phase:.6f}"/>
        <Param name="Wave_Period" value="{wave_period:.2f}"/>"""

print(xml_snippet)