import subprocess

# Plik wynikowy
REPORT_FILE = "raport_poprawiony.txt"

def run_analysis(command, case_name):
    # Wywołanie skryptu analitycznego i przechwycenie wyjścia
    print(f"\n--- [ANALIZA] Przetwarzanie: {case_name} ---")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Przekazanie strumienia na ekran
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    # Zapis wyników do raportu zbiorczego
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"RAPORT DLA WARIANTU: {case_name}\n")
        f.write(f"{'='*50}\n")
        if result.stdout:
            f.write(result.stdout)
        if result.returncode != 0:
            f.write(f"\n[BŁĄD] Analiza zakończona kodem {result.returncode}\n")
            if result.stderr:
                f.write(result.stderr)

def main():
    # Inicjalizacja pliku raportu
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== POPRAWIONY RAPORT Z ANALIZY WSPÓŁCZYNNIKÓW FALOWYCH ===\n")

    # Wariant 1: Kanał gładki (odczyt z output01)
    cmd_clean = "python3 kp/shallow_water/thesis/anal_complete_clean_canal.py output01/"
    run_analysis(cmd_clean, "01_CLEAN_CANAL")

    # Wariant 2: Puste wnęki (odczyt z output02)
    cmd_empty = "python3 kp/shallow_water/thesis/anal_complete.py 02empty_cav_VTK_P00_ output02/"
    run_analysis(cmd_empty, "02_EMPTY_CAVITY")

    # Wariant 3: Wnęki z żebrami (odczyt z output03)
    cmd_fins = "python3 kp/shallow_water/thesis/anal_complete.py 03fins_VTK_P00_ output03/"
    run_analysis(cmd_fins, "03_FIN_CAVITY")
    
    print(f"\n[System] Zakończono re-analizę. Wyniki zapisano do: {REPORT_FILE}")

if __name__ == "__main__":
    main()