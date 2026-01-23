@echo off
echo ============================================
echo ENSITECH - Systeme de Controle Vestimentaire
echo ============================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer Python 3.8+ depuis https://python.org
    pause
    exit /b 1
)

REM Créer un environnement virtuel si nécessaire
if not exist "venv" (
    echo Creation de l'environnement virtuel...
    python -m venv venv
)

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Installer les dépendances
echo Installation des dependances...
pip install -r requirements.txt

REM Créer le dossier alerts
if not exist "alerts" mkdir alerts

echo.
echo ============================================
echo Demarrage du serveur...
echo Interface disponible sur: http://localhost:5000
echo ============================================
echo.

REM Lancer l'application
python app.py

pause
