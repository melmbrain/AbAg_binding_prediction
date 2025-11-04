@echo off
REM Start Background Embedding Generation with Checkpoint System
REM This script starts ESM2 embedding generation in background

echo ================================================================================
echo           STARTING BACKGROUND EMBEDDING GENERATION
echo ================================================================================
echo.
echo This will generate ESM2 embeddings for:
echo   - 185,718 AbBiBench samples
echo   -      53 Therapeutic antibody samples
echo   ------------------------------------------------
echo   - 185,771 total samples
echo.
echo Configuration:
echo   - Mode: CPU (zero GPU conflict)
echo   - Batch size: 16 sequences
echo   - Checkpoint: Every 50 batches
echo   - Auto-resume: YES
echo.

REM Check if checkpoint already exists
if exist external_data\embedding_checkpoint.pkl (
    echo [INFO] Found existing checkpoint - will resume from previous progress
    echo.

    REM Show progress
    python -c "import pickle; cp = pickle.load(open('external_data/embedding_checkpoint.pkl', 'rb')); print(f'Previous progress: {cp[\"last_index\"]:,} / 185,771 ({cp.get(\"progress_pct\", 0):.1f}%%)')"
    echo.

    set /p resume="Resume from checkpoint? (Y/N): "
    if /i "%resume%"=="N" (
        echo [INFO] Removing old checkpoint...
        del external_data\embedding_checkpoint.pkl
        echo [OK] Starting fresh
    ) else (
        echo [OK] Resuming from checkpoint
    )
) else (
    echo [INFO] No checkpoint found - starting fresh
)

echo.
echo ================================================================================
echo Starting embedding generation in background...
echo ================================================================================
echo.
echo The process will run in background. You can:
echo   - Check progress: python scripts/check_embedding_progress.py
echo   - View log:      type embedding_generation.log
echo   - Stop process:  taskkill /F /IM python.exe /FI "WINDOWTITLE eq Embedding*"
echo.

REM Start in background and redirect output to log
start "Embedding Generation" /B python.exe scripts/generate_embeddings_incremental.py --use_cpu --batch_size 16 --save_every 50 > embedding_generation.log 2>&1

echo.
echo [OK] Embedding generation started in background
echo [INFO] Log file: embedding_generation.log
echo [INFO] Checkpoint file: external_data/embedding_checkpoint.pkl
echo.
echo Estimated completion time: 1-2 days
echo.
echo Check progress with: python scripts/check_embedding_progress.py
echo.

pause
