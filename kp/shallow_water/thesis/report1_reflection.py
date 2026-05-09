import sys
import pandas as pd
import matplotlib
# Wymuszenie trybu bezokienkowego do zapisu plików.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pobranie argumentów wejściowych ze skryptu.
plik_csv = sys.argv[1]
nazwa_kolumny = sys.argv[2]

# Wczytanie danych wejściowych z pliku CSV.
df = pd.read_csv(plik_csv)

# Rysowanie wykresu liniowego dla zdefiniowanej kolumny.
plt.plot(df[nazwa_kolumny])

# Zapis wyniku działania do pliku PNG.
plt.savefig('wykres.png')