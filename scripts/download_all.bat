@echo off
REM Master script to download all priority databases (Windows version)
REM Run this from Windows Command Prompt or PowerShell

echo ================================================================================
echo DOWNLOADING ALL PRIORITY DATABASES
echo ================================================================================
echo.
echo This script will download:
echo   1. AbBiBench (from Hugging Face)
echo   2. SAAINT-DB (from GitHub)
echo   3. PDBbind (requires manual download - instructions will be shown)
echo.
echo ================================================================================
echo.

REM Create external_data directory
if not exist external_data mkdir external_data

REM 1. Download AbBiBench
echo.
echo ================================================================================
echo 1. DOWNLOADING ABBIBENCH
echo ================================================================================
echo.

if exist external_data\abbibench_raw.csv (
    echo [OK] AbBiBench already downloaded ^(abbibench_raw.csv exists^)
    echo   Skipping download...
) else (
    echo Running: python scripts\download_abbibench.py
    python scripts\download_abbibench.py
)

REM 2. Download SAAINT-DB
echo.
echo ================================================================================
echo 2. DOWNLOADING SAAINT-DB
echo ================================================================================
echo.

if exist external_data\saaint_raw.csv (
    echo [OK] SAAINT-DB already downloaded ^(saaint_raw.csv exists^)
    echo   Skipping download...
) else (
    echo Running: python scripts\download_saaint.py
    python scripts\download_saaint.py
)

REM 3. PDBbind (manual download required)
echo.
echo ================================================================================
echo 3. PDBBIND (MANUAL DOWNLOAD REQUIRED)
echo ================================================================================
echo.

if exist external_data\pdbbind_raw.csv (
    echo [OK] PDBbind already downloaded ^(pdbbind_raw.csv exists^)
    echo   Skipping download...
) else (
    echo [WARNING] PDBbind requires manual download from website
    echo.
    echo Please follow these steps:
    echo   1. Visit: http://www.pdbbind.org.cn/download.php
    echo   2. Download: PP_INDEX_general_set.2020
    echo   3. Save to: external_data\
    echo   4. Run: python scripts\download_pdbbind.py
    echo.
    echo Or register for 2024 version:
    echo   1. Visit: https://www.pdbbind-plus.org.cn/
    echo   2. Register ^(free for academics^)
    echo   3. Download protein-protein index
    echo   4. Save to: external_data\
    echo   5. Run: python scripts\download_pdbbind.py
    echo.
    echo You can continue with AbBiBench and SAAINT-DB for now.
)

REM Summary
echo.
echo ================================================================================
echo DOWNLOAD SUMMARY
echo ================================================================================
echo.

dir external_data\*.csv 2>nul || echo No CSV files found yet

echo.
echo Next steps:
echo   1. Download PDBbind (if not done)
echo   2. Review downloaded data files
echo   3. Run integration:
echo      python scripts\integrate_all_databases.py --existing YOUR_EXISTING_DATA.csv
echo.
echo ================================================================================

pause
