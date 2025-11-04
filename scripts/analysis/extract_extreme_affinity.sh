#!/bin/bash

echo "================================================================================"
echo "EXTRACTING EXTREME AFFINITY DATA FROM DATABASES"
echo "================================================================================"

BASE_PATH="/mnt/c/Users/401-24/Desktop/Docking prediction"
OUTPUT_DIR="/mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/extreme_affinity_data"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "1. Extracting VERY STRONG binders (pKd > 11) from SKEMPI2"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/raw/affinity_databases/skempi2.csv"

# Extract header
head -1 "$FILE" | sed 's/;/,/g' > "$OUTPUT_DIR/skempi2_very_strong.csv"

# Extract very strong binders (Kd < 1e-11 M, which is pKd > 11)
tail -n +2 "$FILE" | awk -F';' '{
    # Column 10 is Affinity_wt
    affinity = $10
    if (affinity != "" && affinity > 0 && affinity < 1e-11) {
        print $0
    }
}' | sed 's/;/,/g' >> "$OUTPUT_DIR/skempi2_very_strong.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/skempi2_very_strong.csv" | wc -l)
echo "Extracted $COUNT very strong binders (pKd > 11)"

echo ""
echo "2. Extracting VERY WEAK binders (pKd < 5) from SKEMPI2"
echo "--------------------------------------------------------------------------------"
# Extract header
head -1 "$FILE" | sed 's/;/,/g' > "$OUTPUT_DIR/skempi2_very_weak.csv"

# Extract very weak binders (Kd > 1e-5 M, which is pKd < 5)
tail -n +2 "$FILE" | awk -F';' '{
    # Column 10 is Affinity_wt
    affinity = $10
    if (affinity != "" && affinity > 1e-5) {
        print $0
    }
}' | sed 's/;/,/g' >> "$OUTPUT_DIR/skempi2_very_weak.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/skempi2_very_weak.csv" | wc -l)
echo "Extracted $COUNT very weak binders (pKd < 5)"

echo ""
echo "3. Extracting WEAK binders (pKd 5-7) from SKEMPI2"
echo "--------------------------------------------------------------------------------"
# Extract header
head -1 "$FILE" | sed 's/;/,/g' > "$OUTPUT_DIR/skempi2_weak.csv"

# Extract weak binders (Kd 1e-7 to 1e-5 M, which is pKd 5-7)
tail -n +2 "$FILE" | awk -F';' '{
    # Column 10 is Affinity_wt
    affinity = $10
    if (affinity != "" && affinity >= 1e-7 && affinity <= 1e-5) {
        print $0
    }
}' | sed 's/;/,/g' >> "$OUTPUT_DIR/skempi2_weak.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/skempi2_weak.csv" | wc -l)
echo "Extracted $COUNT weak binders (pKd 5-7)"

echo ""
echo "4. Extracting STRONG binders (pKd 9-11) from SKEMPI2"
echo "--------------------------------------------------------------------------------"
# Extract header
head -1 "$FILE" | sed 's/;/,/g' > "$OUTPUT_DIR/skempi2_strong.csv"

# Extract strong binders (Kd 1e-11 to 1e-9 M, which is pKd 9-11)
tail -n +2 "$FILE" | awk -F';' '{
    # Column 10 is Affinity_wt
    affinity = $10
    if (affinity != "" && affinity >= 1e-11 && affinity <= 1e-9) {
        print $0
    }
}' | sed 's/;/,/g' >> "$OUTPUT_DIR/skempi2_strong.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/skempi2_strong.csv" | wc -l)
echo "Extracted $COUNT strong binders (pKd 9-11)"

echo ""
echo "5. Extracting extreme binders from SAbDab"
echo "--------------------------------------------------------------------------------"
FILE="$BASE_PATH/data/raw/affinity_databases/sabdab/sabdab_affinity_data.csv"

# Very strong (pKd > 11)
head -1 "$FILE" > "$OUTPUT_DIR/sabdab_very_strong.csv"
tail -n +2 "$FILE" | awk -F',' '{
    pkd = $3
    if (pkd != "" && pkd > 11) {
        print $0
    }
}' >> "$OUTPUT_DIR/sabdab_very_strong.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/sabdab_very_strong.csv" | wc -l)
echo "Extracted $COUNT very strong binders (pKd > 11) from SAbDab"

# Very weak (pKd < 5)
head -1 "$FILE" > "$OUTPUT_DIR/sabdab_very_weak.csv"
tail -n +2 "$FILE" | awk -F',' '{
    pkd = $3
    if (pkd != "" && pkd < 5) {
        print $0
    }
}' >> "$OUTPUT_DIR/sabdab_very_weak.csv"

COUNT=$(tail -n +2 "$OUTPUT_DIR/sabdab_very_weak.csv" | wc -l)
echo "Extracted $COUNT very weak binders (pKd < 5) from SAbDab"

echo ""
echo "================================================================================"
echo "IDENTIFYING ANTIBODY-ANTIGEN COMPLEXES"
echo "================================================================================"
echo ""
echo "Checking for antibody/immunoglobulin keywords in SKEMPI2 very strong binders:"
echo "--------------------------------------------------------------------------------"

# Check protein names in SKEMPI2 for antibody-related keywords
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_very_strong.csv" | wc -l | awk '{print "Found " $1 " antibody-antigen complexes"}'

echo ""
echo "Sample antibody-antigen complexes (very strong):"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_very_strong.csv" | head -5 | cut -d',' -f1-4

echo ""
echo "Checking for antibody keywords in SKEMPI2 weak binders:"
echo "--------------------------------------------------------------------------------"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_weak.csv" | wc -l | awk '{print "Found " $1 " antibody-antigen complexes"}'

echo ""
echo "Sample antibody-antigen complexes (weak):"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_weak.csv" | head -5 | cut -d',' -f1-4

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "Extracted files saved to: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"/*.csv | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Next steps:"
echo "1. Review the antibody-antigen complexes in the extracted files"
echo "2. Add these to your training data to improve model performance on extremes"
echo "3. Prioritize:"
echo "   - Very strong binders (pKd > 11) - almost none in current dataset"
echo "   - Weak/very weak binders (pKd < 7) - underrepresented in current dataset"
echo ""
