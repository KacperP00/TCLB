import glob

# Skrypt do lokalnego łączenia wyników w zbiorcze pliki tekstowe dla AI.
# Należy uruchomić w folderze z pobranymi wynikami.

def merge_files(pattern, output_file):
    # Wyszukiwanie plików według wzorca i alfabetyczne sortowanie
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Brak plików dla wzorca: {pattern}")
        return

    # Zapis zawartości do jednego pliku z nagłówkami separującymi
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in files:
            outfile.write(f"\n{'='*60}\n--- PLIK ZRODLOWY: {fname} ---\n{'='*60}\n")
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write("\n")
    
    print(f"Złączono {len(files)} plików -> {output_file}")

if __name__ == "__main__":
    # Agregacja raportów R/T (8 plików txt)
    merge_files("*_report.txt", "MASTER_RAPORT_RT.txt")
    
    # Agregacja konfiguracji XML (8 plików xml zapisanych jako txt dla AI)
    merge_files("xml/*.xml", "MASTER_CONFIGS.txt")