import os
import subprocess
import numpy as np

# 1. Definicja badanej przestrzeni (Zmienione na 2 iteracje dla szybkiego testu!)
# Gdy upewnisz się, że plik CSV się buduje, zmień '2' na '20'
lambdas = np.linspace(200, 400, 20)

for l in lambdas:
    print(f"\n==========================================")
    print(f"   ROZPOCZYNAM ITERACJE: lambda = {l:.1f}")
    print(f"==========================================\n")
    
    print("0/3 -> Sprzątanie po poprzedniej symulacji...")
    subprocess.run(["rm", "-rf", "output/"], check=False)

    # 2. Wywołanie skryptu do generacji fali (zapisze plik .csv dla TCLB)
    print("1/3 -> Generowanie pliku fali wejściowej...")
    subprocess.run(["python3", "kp/shallow_water/thesis/inlet_wave.py", str(l)], check=True)
    
    # 3. Uruchomienie symulacji TCLB (Czeka na zakończenie dzięki --wait)
    print("2/3 -> Symulacja TCLB na GPU...")
    subprocess.run(["./p/run", "sw", "kp/shallow_water/thesis/clean_canal.xml", "1", "--wait"], check=True)
    
    # 4. Analiza wyników i dopisanie (append) do zbiorczego pliku
    print("3/3 -> Analiza wyników (baseline.py)...")
    subprocess.run(["python3", "kp/shallow_water/thesis/baseline.py", str(l)], check=True)
    
print("\n>>> EKSPERYMENT ZAKONCZONY POMYSLNIE <<<")