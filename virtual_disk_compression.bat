@echo off
chcp 65001 > nul
echo ==========================================
echo Kompresja dysku WSL 2 (Ubuntu)
echo ==========================================

:: Sprawdzenie czy skrypt ma uprawnienia administratora
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [BŁĄD] Narzędzie Diskpart wymaga uprawnień administratora!
    echo Zamknij to okno, kliknij prawym przyciskiem myszy na skrót i wybierz "Uruchom jako administrator".
    echo.
    pause
    exit /b 1
)

echo [1/3] Zamykanie WSL...
wsl --shutdown
if %errorlevel% neq 0 (
    echo [BŁĄD] Nie udało się zamknąć WSL.
    pause
    exit /b 1
)
echo [SUKCES] WSL zamknięty.

echo [2/3] Przygotowywanie instrukcji dla Diskpart...
set DPT_SCRIPT="%TEMP%\wsl_compact.txt"
echo select vdisk file="C:\Users\piech\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx" > %DPT_SCRIPT%
echo compact vdisk >> %DPT_SCRIPT%
echo exit >> %DPT_SCRIPT%

echo [3/3] Kompresowanie dysku (może to chwilę potrwać, proszę czekać)...
diskpart /s %DPT_SCRIPT%
set DP_ERROR=%errorlevel%

:: Czyszczenie pliku tymczasowego
del %DPT_SCRIPT%

if %DP_ERROR% neq 0 (
    echo.
    echo [BŁĄD] Operacja Diskpart zakończyła się niepowodzeniem (Kod: %DP_ERROR%).
) else (
    echo.
    echo [SUKCES] Dysk został pomyślnie skompresowany i odzyskałeś wolne miejsce!
)

echo.
pause