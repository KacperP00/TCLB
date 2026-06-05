import re
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do pliku z wynikami analizy
REPORT_FILE = "raport_poprawiony.txt"
PLOT_FILE = "wykresy/bilans_energii.png"

def parse_report(filepath):
    # Wczytanie zawartości pliku tekstowego
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Podział tekstu na bloki poszczególnych wariantów
    blocks = content.split("RAPORT DLA WARIANTU: ")[1:]
    
    names = []
    R_vals = []
    T_vals = []
    
    # Ekstrakcja nazw i wartości liczbowych za pomocą wyrażeń regularnych
    for block in blocks:
        name = block.split("\n")[0].strip()
        names.append(name)
        
        match_R = re.search(r"=> Odbicie \(R\):\s+([0-9.]+)", block)
        match_T = re.search(r"=> Transmisja \(T\):\s+([0-9.]+)", block)
        
        if match_R and match_T:
            R_vals.append(float(match_R.group(1)))
            T_vals.append(float(match_T.group(1)))
            
    return names, np.array(R_vals), np.array(T_vals)

def main():
    # Pobranie danych wejściowych
    names, R, T = parse_report(REPORT_FILE)
    
    # Przeliczenie amplitud na udziały energii
    E_R = R**2
    E_T = T**2
    D = 1.0 - (E_R + E_T)
    
    # Inicjalizacja obszaru roboczego wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie osi X
    x = np.arange(len(names))
    width = 0.6
    
    # Rysowanie słupków skumulowanych
    p1 = ax.bar(x, E_T * 100, width, label='Transmitowana (T²)', color='#2ca02c', edgecolor='black')
    p2 = ax.bar(x, D * 100, width, bottom=E_T * 100, label='Dyssypacja (D)', color='#d62728', edgecolor='black')
    p3 = ax.bar(x, E_R * 100, width, bottom=(E_T + D) * 100, label='Odbita (R²)', color='#1f77b4', edgecolor='black')
    
    # Formatowanie wykresu
    ax.set_ylabel('Udział energii fali padającej [%]', fontsize=12)
    ax.set_title('Bilans energii fali dla różnych wariantów geometrii', fontsize=14, pad=15)
    ax.set_xticks(x)
    
    # Tłumaczenie etykiet
    labels_pl = [n.replace("01_CLEAN_CANAL", "Kanał Gładki")
                  .replace("02_EMPTY_CAVITY", "Puste Wnęki")
                  .replace("03_FIN_CAVITY", "Wnęki z Żebrami") for n in names]
    ax.set_xticklabels(labels_pl, fontsize=11)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dodanie wartości procentowych na słupkach
    for i in range(len(names)):
        ax.text(x[i], E_T[i]*100/2, f"{E_T[i]*100:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        ax.text(x[i], (E_T[i] + D[i]/2)*100, f"{D[i]*100:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        if E_R[i] * 100 > 1.0:
            ax.text(x[i], (E_T[i] + D[i] + E_R[i]/2)*100, f"{E_R[i]*100:.1f}%", ha='center', va='center', color='black', fontweight='bold')
            
    # Zapis i wyświetlenie
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    print(f"[System] Zapisano wykres do pliku: {PLOT_FILE}")

if __name__ == "__main__":
    main()