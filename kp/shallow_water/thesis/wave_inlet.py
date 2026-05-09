import math
import os

def generate_wave():
    # Ścieżka pliku wyjściowego
    file_path = "kp/shallow_water/thesis/inlet_wave.csv"

    # Parametry fizyczne modelu SWE w TCLB
    H_sim = 30.0
    g_sim = 0.00066667
    c_sim = math.sqrt(g_sim * H_sim)

    # Parametry fali wymuszającej
    amplitude = 0.0015
    lambda_target = 500.0
    
    # Obliczenie okresu fali w iteracjach
    period_calc = lambda_target / c_sim
    period = int(round(period_calc))

    # Weryfikacja rzeczywistych parametrów po zaokrągleniu okresu
    real_lambda = period * c_sim

    print("--- KONFIGURACJA FALI ---")
    print(f"Głębokość H={H_sim}, Grawitacja g={g_sim}")
    print(f"Prędkość fali c={c_sim:.5f} lu/iter")
    print(f"Celowana długość fali L={lambda_target} lu")
    print("-" * 30)
    print(f"Wyliczony okres T = {period_calc:.2f}")
    print(f"PRZYJĘTY OKRES T = {period} iteracji")
    print(f"Rzeczywista długość fali L = {real_lambda:.2f} lu")
    print(f"Amplituda = {amplitude:.5f} lu")
    print("-" * 30)
    

    # Długość symulacji zgodna z plikiem konfiguracyjnym XML
    total_iters = 30000

    print(f"Generowanie pliku: {file_path}...")

    # Zapis przebiegu czasowego wymuszenia do pliku CSV
    with open(file_path, "w") as f:
        f.write("iter,cos\n")
        
        for i in range(total_iters + 1):
            # Równanie fali z przesunięciem fazowym -pi/2 (start od 0)
            val = amplitude * math.cos(2 * math.pi * i / period - math.pi/2)
            f.write(f"{i},{val:.6f}\n")

if __name__ == "__main__":
    generate_wave()