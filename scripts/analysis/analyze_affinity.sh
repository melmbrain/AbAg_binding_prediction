#!/bin/bash

echo "================================================================================"
echo "ANALYZING AFFINITY DISTRIBUTION IN CURRENT AND AVAILABLE DATASETS"
echo "================================================================================"

# Base path
BASE_PATH="/mnt/c/Users/401-24/Desktop/Docking prediction"

echo ""
echo "1. CURRENT DATASET (Phase 6 - 205k dataset)"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/processed/phase6/final_205k_dataset.csv"
if [ -f "$FILE" ]; then
    TOTAL=$(tail -n +2 "$FILE" | wc -l)
    echo "Total samples: $TOTAL"
    echo ""
    echo "Analyzing pKd distribution (higher = stronger binding):"

    # Extract pKd values (column 2) and analyze distribution
    tail -n +2 "$FILE" | cut -d',' -f2 | awk '
    BEGIN {
        very_weak=0; weak=0; moderate=0; strong=0; very_strong=0
        min=999; max=-999; sum=0; count=0
    }
    {
        if ($1 != "" && $1 > 0) {
            count++
            sum += $1
            if ($1 < min) min = $1
            if ($1 > max) max = $1

            # pKd classification
            if ($1 < 5) very_weak++
            else if ($1 < 7) weak++
            else if ($1 < 9) moderate++
            else if ($1 < 11) strong++
            else very_strong++
        }
    }
    END {
        if (count > 0) {
            avg = sum/count
            print "  Range: [" min ", " max "]"
            print "  Mean: " avg
            print ""
            print "  Distribution:"
            print "    pKd < 5 (very weak, Kd > 10 μM):     " very_weak " (" int(very_weak*100/count) "%)"
            print "    pKd 5-7 (weak, Kd 100 nM - 10 μM):  " weak " (" int(weak*100/count) "%)"
            print "    pKd 7-9 (moderate, Kd 1-100 nM):    " moderate " (" int(moderate*100/count) "%)"
            print "    pKd 9-11 (strong, Kd 0.01-1 nM):    " strong " (" int(strong*100/count) "%)"
            print "    pKd > 11 (very strong, Kd < 10 pM): " very_strong " (" int(very_strong*100/count) "%)"
        }
    }'
else
    echo "File not found: $FILE"
fi

echo ""
echo ""
echo "2. SABDAB AFFINITY DATABASE"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/raw/affinity_databases/sabdab/sabdab_affinity_data.csv"
if [ -f "$FILE" ]; then
    TOTAL=$(tail -n +2 "$FILE" | wc -l)
    echo "Total samples: $TOTAL"
    echo ""
    echo "Analyzing pKd distribution:"

    tail -n +2 "$FILE" | cut -d',' -f3 | awk '
    BEGIN {
        very_weak=0; weak=0; moderate=0; strong=0; very_strong=0
        min=999; max=-999; sum=0; count=0
    }
    {
        if ($1 != "" && $1 > 0) {
            count++
            sum += $1
            if ($1 < min) min = $1
            if ($1 > max) max = $1

            if ($1 < 5) very_weak++
            else if ($1 < 7) weak++
            else if ($1 < 9) moderate++
            else if ($1 < 11) strong++
            else very_strong++
        }
    }
    END {
        if (count > 0) {
            avg = sum/count
            print "  Range: [" min ", " max "]"
            print "  Mean: " avg
            print ""
            print "  Distribution:"
            print "    pKd < 5 (very weak):     " very_weak " (" int(very_weak*100/count) "%)"
            print "    pKd 5-7 (weak):          " weak " (" int(weak*100/count) "%)"
            print "    pKd 7-9 (moderate):      " moderate " (" int(moderate*100/count) "%)"
            print "    pKd 9-11 (strong):       " strong " (" int(strong*100/count) "%)"
            print "    pKd > 11 (very strong):  " very_strong " (" int(very_strong*100/count) "%)"
        }
    }'
else
    echo "File not found: $FILE"
fi

echo ""
echo ""
echo "3. PPB-AFFINITY DATABASE"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/raw/affinity_databases/ppb_affinity/PPB-Affinity.csv"
if [ -f "$FILE" ]; then
    TOTAL=$(tail -n +2 "$FILE" | wc -l)
    echo "Total samples: $TOTAL"
    echo ""
    echo "Sample of Kd values (in Molar):"
    tail -n +2 "$FILE" | cut -d',' -f10 | grep -E "^[0-9]" | head -20

    echo ""
    echo "Analyzing Kd distribution (converting to pKd):"
    tail -n +2 "$FILE" | cut -d',' -f10 | grep -E "^[0-9]" | awk '
    BEGIN {
        very_weak=0; weak=0; moderate=0; strong=0; very_strong=0
        min=999; max=-999; sum=0; count=0
    }
    {
        if ($1 != "" && $1 > 0) {
            # Convert Kd (M) to pKd = -log10(Kd)
            pkd = -log($1)/log(10)
            count++
            sum += pkd
            if (pkd < min) min = pkd
            if (pkd > max) max = pkd

            if (pkd < 5) very_weak++
            else if (pkd < 7) weak++
            else if (pkd < 9) moderate++
            else if (pkd < 11) strong++
            else very_strong++
        }
    }
    END {
        if (count > 0) {
            avg = sum/count
            print "  Range: [" min ", " max "]"
            print "  Mean: " avg
            print ""
            print "  Distribution:"
            print "    pKd < 5 (very weak):     " very_weak " (" int(very_weak*100/count) "%)"
            print "    pKd 5-7 (weak):          " weak " (" int(weak*100/count) "%)"
            print "    pKd 7-9 (moderate):      " moderate " (" int(moderate*100/count) "%)"
            print "    pKd 9-11 (strong):       " strong " (" int(strong*100/count) "%)"
            print "    pKd > 11 (very strong):  " very_strong " (" int(very_strong*100/count) "%)"
        }
    }'
else
    echo "File not found: $FILE"
fi

echo ""
echo ""
echo "4. SKEMPI2 DATABASE"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/raw/affinity_databases/skempi2.csv"
if [ -f "$FILE" ]; then
    TOTAL=$(tail -n +2 "$FILE" | wc -l)
    echo "Total samples: $TOTAL"
    echo ""
    echo "Analyzing wild-type affinity (Affinity_wt in Molar, column 10):"

    tail -n +2 "$FILE" | cut -d';' -f10 | grep -E "^[0-9]" | awk '
    BEGIN {
        very_weak=0; weak=0; moderate=0; strong=0; very_strong=0
        min=999; max=-999; sum=0; count=0
    }
    {
        if ($1 != "" && $1 > 0) {
            # Convert Kd (M) to pKd = -log10(Kd)
            pkd = -log($1)/log(10)
            count++
            sum += pkd
            if (pkd < min) min = pkd
            if (pkd > max) max = pkd

            if (pkd < 5) very_weak++
            else if (pkd < 7) weak++
            else if (pkd < 9) moderate++
            else if (pkd < 11) strong++
            else very_strong++
        }
    }
    END {
        if (count > 0) {
            avg = sum/count
            print "  Range: [" min ", " max "]"
            print "  Mean: " avg
            print ""
            print "  Distribution:"
            print "    pKd < 5 (very weak):     " very_weak " (" int(very_weak*100/count) "%)"
            print "    pKd 5-7 (weak):          " weak " (" int(weak*100/count) "%)"
            print "    pKd 7-9 (moderate):      " moderate " (" int(moderate*100/count) "%)"
            print "    pKd 9-11 (strong):       " strong " (" int(strong*100/count) "%)"
            print "    pKd > 11 (very strong):  " very_strong " (" int(very_strong*100/count) "%)"
        }
    }'
else
    echo "File not found: $FILE"
fi

echo ""
echo ""
echo "================================================================================"
echo "RECOMMENDATIONS"
echo "================================================================================"
echo ""
echo "Based on the analysis, the following databases should be considered for"
echo "adding EXTREME affinity data (weak and strong):"
echo ""
echo "1. For WEAK BINDERS (pKd < 7, Kd > 100 nM):"
echo "   - Check PPB-Affinity for weak binders"
echo "   - Check SKEMPI2 for weak mutant binders"
echo ""
echo "2. For STRONG BINDERS (pKd > 11, Kd < 10 pM):"
echo "   - Check PPB-Affinity for very strong binders"
echo "   - Check SKEMPI2 for very strong wild-type complexes"
echo ""
