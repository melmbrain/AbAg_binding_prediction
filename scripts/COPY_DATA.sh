#!/bin/bash
# Quick script to copy your data to the project directory

echo "Copying dataset to project..."

# Create data directory
mkdir -p data

# Copy the phase2 dataset (recommended)
cp "/mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv" data/
cp "/mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_sample.csv" data/

echo "✅ Data copied to data/ directory"
echo ""
echo "Files available:"
ls -lh data/*.csv

echo ""
echo "Next step: Upload these files to Google Drive in folder 'AbAg_data'"
