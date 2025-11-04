#!/bin/bash

echo "================================================================================"
echo "ANTIBODY-ANTIGEN EXTREME AFFINITY DATA SUMMARY"
echo "================================================================================"

OUTPUT_DIR="/mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/extreme_affinity_data"
SUMMARY_FILE="$OUTPUT_DIR/antibody_antigen_summary.txt"

> "$SUMMARY_FILE"  # Clear file

echo ""
echo "1. VERY STRONG ANTIBODY-ANTIGEN BINDERS (pKd > 11)"
echo "================================================================================"
echo ""
echo "From SAbDab (5 antibody-antigen complexes):"
echo "--------------------------------------------------------------------------------"
echo "PDB_Code  Kd (nM)      pKd     Original_Value  Source" | tee -a "$SUMMARY_FILE"
echo "--------  ---------  -------  --------------  ------" | tee -a "$SUMMARY_FILE"

tail -n +2 "$OUTPUT_DIR/sabdab_very_strong.csv" | awk -F',' '{
    if (NF > 0) {
        printf("%-8s  %9.5f  %7.2f  %-14s  %s\n", $1, $2, $3, $4, $5)
    }
}' | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "From SKEMPI2 (antibody-related only):"
echo "--------------------------------------------------------------------------------"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_very_strong.csv" > "$OUTPUT_DIR/skempi2_antibody_very_strong.csv"
COUNT=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_very_strong.csv")
echo "Found $COUNT antibody-related very strong binders" | tee -a "$SUMMARY_FILE"

if [ $COUNT -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Sample entries:" | tee -a "$SUMMARY_FILE"
    head -5 "$OUTPUT_DIR/skempi2_antibody_very_strong.csv" | cut -d',' -f1,12,13,9,10 | awk -F',' '{
        printf("  PDB: %-15s  Protein1: %-30s  Protein2: %-30s\n", $1, $2, $3)
        if ($4 != "") {
            # Convert Kd (M) to pKd
            kd_m = $4
            if (kd_m > 0) {
                pkd = -log(kd_m)/log(10)
                printf("    Kd_wt: %s M (pKd: %.2f)\n", $4, pkd)
            }
        }
    }' | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "2. WEAK/VERY WEAK ANTIBODY-ANTIGEN BINDERS (pKd < 7)"
echo "================================================================================"
echo ""
echo "From SAbDab:"
echo "--------------------------------------------------------------------------------"
echo "PDB_Code  Kd (nM)      pKd     Original_Value  Source" | tee -a "$SUMMARY_FILE"
echo "--------  ---------  -------  --------------  ------" | tee -a "$SUMMARY_FILE"

tail -n +2 "$OUTPUT_DIR/sabdab_very_weak.csv" | awk -F',' '{
    if (NF > 0) {
        printf("%-8s  %9.2f  %7.2f  %-14s  %s\n", $1, $2, $3, $4, $5)
    }
}' | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "From SKEMPI2 weak binders (antibody-related only):"
echo "--------------------------------------------------------------------------------"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_weak.csv" > "$OUTPUT_DIR/skempi2_antibody_weak.csv"
COUNT=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_weak.csv")
echo "Found $COUNT antibody-related weak binders" | tee -a "$SUMMARY_FILE"

if [ $COUNT -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Sample entries:" | tee -a "$SUMMARY_FILE"
    head -10 "$OUTPUT_DIR/skempi2_antibody_weak.csv" | cut -d',' -f1,12,13,9,10 | awk -F',' '{
        printf("  PDB: %-15s  Protein1: %-30s  Protein2: %-30s\n", $1, $2, $3)
        if ($4 != "") {
            # Convert Kd (M) to pKd
            kd_m = $4
            if (kd_m > 0) {
                pkd = -log(kd_m)/log(10)
                printf("    Kd_wt: %s M (pKd: %.2f)\n", $4, pkd)
            }
        }
    }' | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "From SKEMPI2 very weak binders (antibody-related only):"
echo "--------------------------------------------------------------------------------"
grep -i -E "antibod|immunoglobulin|fab|scfv|igg|nanobody|vhh" "$OUTPUT_DIR/skempi2_very_weak.csv" > "$OUTPUT_DIR/skempi2_antibody_very_weak.csv"
COUNT=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_very_weak.csv")
echo "Found $COUNT antibody-related very weak binders" | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================================"
echo "COMPREHENSIVE SUMMARY"
echo "================================================================================"
echo "" | tee -a "$SUMMARY_FILE"

echo "ANTIBODY-ANTIGEN COMPLEXES AVAILABLE FOR ADDING TO YOUR DATASET:" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Very strong
VS_SABDAB=$(tail -n +2 "$OUTPUT_DIR/sabdab_very_strong.csv" | wc -l)
VS_SKEMPI2=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_very_strong.csv")
TOTAL_VS=$((VS_SABDAB + VS_SKEMPI2))

# Weak
WEAK_SABDAB=$(tail -n +2 "$OUTPUT_DIR/sabdab_very_weak.csv" | wc -l)
WEAK_SKEMPI2=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_weak.csv")
VW_SKEMPI2=$(wc -l < "$OUTPUT_DIR/skempi2_antibody_very_weak.csv")
TOTAL_WEAK=$((WEAK_SABDAB + WEAK_SKEMPI2 + VW_SKEMPI2))

echo "1. VERY STRONG BINDERS (pKd > 11):" | tee -a "$SUMMARY_FILE"
echo "   - SAbDab:           $VS_SABDAB complexes" | tee -a "$SUMMARY_FILE"
echo "   - SKEMPI2:          $VS_SKEMPI2 complexes" | tee -a "$SUMMARY_FILE"
echo "   - TOTAL:            $TOTAL_VS complexes" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "2. WEAK/VERY WEAK BINDERS (pKd < 7):" | tee -a "$SUMMARY_FILE"
echo "   - SAbDab:           $WEAK_SABDAB complexes" | tee -a "$SUMMARY_FILE"
echo "   - SKEMPI2 weak:     $WEAK_SKEMPI2 complexes" | tee -a "$SUMMARY_FILE"
echo "   - SKEMPI2 v.weak:   $VW_SKEMPI2 complexes" | tee -a "$SUMMARY_FILE"
echo "   - TOTAL:            $TOTAL_WEAK complexes" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "RECOMMENDATION:" | tee -a "$SUMMARY_FILE"
echo "---------------" | tee -a "$SUMMARY_FILE"
echo "Your current dataset has:" | tee -a "$SUMMARY_FILE"
echo "  - Only 0.1% very strong binders (pKd > 11) = 240 out of 205k" | tee -a "$SUMMARY_FILE"
echo "  - Only 1.8% very weak binders (pKd < 5) = 3,778 out of 205k" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Priority additions:" | tee -a "$SUMMARY_FILE"
echo "  1. Add ALL $TOTAL_VS very strong Ab-Ag complexes from SAbDab" | tee -a "$SUMMARY_FILE"
echo "  2. Add antibody-antigen weak binders from SKEMPI2 ($WEAK_SKEMPI2 complexes)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Files created:" | tee -a "$SUMMARY_FILE"
echo "  - $OUTPUT_DIR/skempi2_antibody_very_strong.csv" | tee -a "$SUMMARY_FILE"
echo "  - $OUTPUT_DIR/skempi2_antibody_weak.csv" | tee -a "$SUMMARY_FILE"
echo "  - $OUTPUT_DIR/skempi2_antibody_very_weak.csv" | tee -a "$SUMMARY_FILE"
echo "  - $OUTPUT_DIR/sabdab_very_strong.csv" | tee -a "$SUMMARY_FILE"
echo "  - $OUTPUT_DIR/sabdab_very_weak.csv" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "Summary saved to: $SUMMARY_FILE"
