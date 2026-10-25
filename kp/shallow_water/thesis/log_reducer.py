import pandas as pd
import sys
from pathlib import Path

def clean_tclb_log(file_path):
    # Konwersja wejściowej ścieżki znakowej na obiekt Path
    path = Path(file_path)
    
    # Generowanie ścieżki wyjściowej z modyfikacją nazwy pliku
    output_path = path.parent / f"clean_{path.name}"
    
    print(f"Przetwarzanie: {path}...")
    df = pd.read_csv(path)
    initial_rows, initial_cols = df.shape
    
    # Zachowanie pojedynczego (ostatniego) wiersza dla każdej ewaluacji
    df = df.drop_duplicates(subset=['Optimization'], keep='last').copy()
    
    # Identyfikacja kolumn do usunięcia
    cols_to_drop = []
    for col in df.columns:
        col_str = str(col)
        # Odrzucanie duplikatów w jednostkach SI
        if col_str.endswith('_si'):
            cols_to_drop.append(col)
        # Odrzucanie pochodnych Adjoint globalnych parametrów
        elif col_str.endswith('_D'):
            cols_to_drop.append(col)
        # Odrzucanie kolumn stale zerowych
        elif (df[col] == 0.0).all():
            cols_to_drop.append(col)
        # Odrzucanie kolumn o stałej wartości niezależnie od przebiegu
        elif df[col].nunique(dropna=False) <= 1:
            cols_to_drop.append(col)
            
    # Wyłączenie kluczowych zmiennych z listy do usunięcia
    safe_to_keep = ['Optimization', 'Iteration', 'Objective', 'WaveError', 'WaveError2', 'Material', 'Penalty']
    cols_to_drop = [c for c in cols_to_drop if c not in safe_to_keep]
    
    # Aplikacja filtrów kolumn
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Odrzucanie rozbicia strefowego dla błędów
    inobj_cols = [c for c in df.columns if 'InObj' in str(c)]
    df.drop(columns=inobj_cols, inplace=True, errors='ignore')

    # Zapis przetworzonej struktury
    df.to_csv(output_path, index=False)
    
    final_rows, final_cols = df.shape
    print(f"Zredukowano wiersze: {initial_rows} -> {final_rows}")
    print(f"Zredukowano kolumny: {initial_cols} -> {final_cols}")
    print(f"Zapisano pomyślnie do: {output_path}")

if __name__ == "__main__":
    # Weryfikacja podanego argumentu w wierszu poleceń
    if len(sys.argv) != 2:
        print("Błąd. Prawidłowe użycie: python skrypt.py <sciezka_do_pliku.csv>")
        sys.exit(1)
        
    clean_tclb_log(sys.argv[1])