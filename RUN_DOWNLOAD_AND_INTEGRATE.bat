@echo off
REM Complete download and integration workflow
REM This script runs all steps automatically

echo ================================================================================
echo COMPLETE DOWNLOAD AND INTEGRATION WORKFLOW
echo ================================================================================
echo.
echo This will:
echo   Step 1: Download AbBiBench and SAAINT-DB
echo   Step 2: Integrate with your existing dataset
echo   Step 3: Generate integration report
echo.
echo Your existing dataset path:
echo   C:\Users\401-24\Desktop\Docking prediction\data\processed\phase6\final_205k_dataset.csv
echo.
echo Press Ctrl+C to cancel, or
pause

REM ================================================================================
REM STEP 1: DOWNLOAD DATABASES
REM ================================================================================

echo.
echo ================================================================================
echo STEP 1: DOWNLOADING DATABASES
echo ================================================================================
echo.

call scripts\download_all.bat

REM ================================================================================
REM STEP 2: INTEGRATE WITH EXISTING DATA
REM ================================================================================

echo.
echo ================================================================================
echo STEP 2: INTEGRATING WITH EXISTING DATASET
echo ================================================================================
echo.

REM Check if existing dataset exists
set EXISTING_DATA=C:\Users\401-24\Desktop\Docking prediction\data\processed\phase6\final_205k_dataset.csv

if not exist "%EXISTING_DATA%" (
    echo [ERROR] Existing dataset not found at:
    echo   %EXISTING_DATA%
    echo.
    echo Please check the path and update this script if needed.
    pause
    exit /b 1
)

echo Running integration...
echo.

python scripts\integrate_all_databases.py ^
  --existing "%EXISTING_DATA%" ^
  --external_dir external_data ^
  --output external_data\merged_all_databases.csv ^
  --report external_data\integration_report.txt

REM ================================================================================
REM STEP 3: SHOW RESULTS
REM ================================================================================

echo.
echo ================================================================================
echo STEP 3: INTEGRATION RESULTS
echo ================================================================================
echo.

if exist external_data\integration_report.txt (
    echo [OK] Integration complete!
    echo.
    echo Report:
    type external_data\integration_report.txt
    echo.
    echo.
    echo Output files:
    dir external_data\merged_all_databases.csv
    echo.
) else (
    echo [WARNING] Integration report not found.
    echo Check console output above for errors.
)

echo ================================================================================
echo COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo   1. Review: external_data\integration_report.txt
echo   2. Check merged data: external_data\merged_all_databases.csv
echo   3. Generate ESM2 embeddings for new sequences
echo   4. Train model with: python train_balanced.py --data external_data\merged_all_databases.csv
echo.

pause
