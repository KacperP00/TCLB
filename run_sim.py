import subprocess
import os
import shutil

# Plik docelowy ze zbiorczym raportem
REPORT_FILE = "raport_koncowy.txt"

def run_simulation(command):
    # Uruchomienie solvera z bezpośrednim wydrukiem postępu na ekran
    print(f"\n--- [SYMULACJA] Rozpoczynam: {command} ---")
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"[Błąd] Symulacja przerwana (Kod: {result.returncode}).")
        return False
    return True

def run_analysis(command, case_name):
    # Uruchomienie analizy, przechwycenie wyniku i zapis do pliku
    print(f"\n--- [ANALIZA] Przetwarzanie danych dla: {case_name} ---")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Wydruk na ekran
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    # Zapis do raportu zbiorczego
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"RAPORT DLA WARIANTU: {case_name}\n")
        f.write(f"{'='*50}\n")
        if result.stdout:
            f.write(result.stdout)
        if result.returncode != 0:
            f.write(f"\n[BŁĄD KRYTYCZNY] Analiza zakończona kodem {result.returncode}\n")
            if result.stderr:
                f.write(result.stderr)

def manage_output(suffix):
    # Zmiana nazwy katalogu po przeprowadzeniu analizy
    base_output = "output"
    new_output = f"output{suffix}"

    if os.path.exists(base_output):
        if os.path.exists(new_output):
            shutil.rmtree(new_output)
            
        os.rename(base_output, new_output)
        print(f"[System] Przeniesiono wyniki do '{new_output}'.")
    
    os.makedirs(base_output, exist_ok=True)

def main():
    # Inicjalizacja czystego pliku raportu
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== ZBIORCZY RAPORT Z ANALIZY WSPÓŁCZYNNIKÓW FALOWYCH ===\n")

    # [1] KANAŁ GŁADKI
    if run_simulation("CLB/sw/main kp/shallow_water/thesis/prep/01clean_canal.xml"):
        run_analysis("python3 kp/shallow_water/thesis/anal_complete_clean_canal.py", "01_CLEAN_CANAL")
        manage_output("01")
    else:
        return

    # [2] PUSTE WNĘKI
    if run_simulation("CLB/sw/main kp/shallow_water/thesis/prep/02empty_cav.xml"):
        run_analysis("python3 kp/shallow_water/thesis/anal_complete.py 02empty_cav_VTK_P00_", "02_EMPTY_CAVITY")
        manage_output("02")
    else:
        return

    # [3] WNĘKI Z ŻEBRAMI
    if run_simulation("CLB/sw/main kp/shallow_water/thesis/prep/03fins.xml"):
        run_analysis("python3 kp/shallow_water/thesis/anal_complete.py 03fins_VTK_P00_", "03_FIN_CAVITY")
        manage_output("03")
    
    print(f"\n[System] Zakończono automatyzację. Wyniki analityczne znajdują się w pliku: {REPORT_FILE}")

if __name__ == "__main__":
    main()