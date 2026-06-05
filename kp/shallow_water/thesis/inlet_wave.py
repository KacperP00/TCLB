import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# --- PARAMETRY WEJŚCIOWE ---
GRAVITY = 0.00209424
HEIGHT = 9.55
AMPLITUDE = 0.0625

# Odczyt długości fali (lambda) z argumentu skryptu
if len(sys.argv) > 1:
    WAVE_LENGTH = float(sys.argv[1])
else:
    WAVE_LENGTH = 400.0 

# --- OBLICZENIA FIZYCZNE ---
c = np.sqrt(GRAVITY * HEIGHT)
period = WAVE_LENGTH / c
omega = 2.0 * np.pi / period

# --- GENEROWANIE FALI (SZTYWNY BUFOR NA 35000 ITERACJI) ---
# Skoro symulacja ma 31113 iteracji, 35000 to idealny zapas.
t = np.arange(35000)
wave_h = AMPLITUDE * np.cos(omega * t)

df = pd.DataFrame({'iter': t, 'cos': wave_h})
df.to_csv('kp/shallow_water/thesis/inlet_wave.csv', index=False)

# --- AUTOMATYCZNA AKTUALIZACJA XML ---
XML_FILE = 'kp/shallow_water/thesis/clean_canal.xml'
try:
    tree = ET.parse(XML_FILE)
    root = tree.getroot()
    # Podmiana wartości w XML, aby TCLB znał bieżące parametry
    for param in root.findall(".//Param"):
        if param.get("name") == "Wave_Period":
            param.set("value", f"{period:.6f}")
        elif param.get("name") == "Wave_Length":
            param.set("value", f"{WAVE_LENGTH:.6f}")
            
    tree.write(XML_FILE, encoding="utf-8", xml_declaration=True)
    print(f"Zaktualizowano {XML_FILE}: lambda = {WAVE_LENGTH}, period = {period:.1f}")
except Exception as e:
    print(f"Błąd! Nie udało się zaktualizować pliku XML: {e}")

print(f"Predkosc fali (c): {c:.6f} lu/tu")
print(f"Czestosc omega: {omega:.6f} 1/tu")
print("Wygenerowano plik CSV z buforem 35000 iteracji.")