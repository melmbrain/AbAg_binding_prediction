#!/bin/bash

echo "=========================================="
echo "AbAg Repository Cleanup for GitHub"
echo "=========================================="
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p archive/old_versions
mkdir -p archive/old_docs  
mkdir -p scripts/analysis
mkdir -p models
echo "✅ Directories created"
echo ""

# Move old training notebooks
echo "Moving old versions..."
if [ -f "colab_training.ipynb" ]; then
    mv colab_training.ipynb archive/old_versions/
    echo "  ✅ Moved colab_training.ipynb (v1) to archive"
fi

# Move old documentation
if [ -f "README_COMPLETE.md" ]; then
    mv README_COMPLETE.md archive/old_docs/
    echo "  ✅ Moved README_COMPLETE.md to archive"
fi

if [ -f "STATUS_AND_NEXT_STEPS.md" ]; then
    mv STATUS_AND_NEXT_STEPS.md archive/old_docs/
    echo "  ✅ Moved STATUS_AND_NEXT_STEPS.md to archive"
fi

if [ -f "TRAINING_RESULTS_ANALYSIS.md" ]; then
    mv TRAINING_RESULTS_ANALYSIS.md docs/reports/v1_analysis.md
    echo "  ✅ Moved TRAINING_RESULTS_ANALYSIS.md to docs/reports/"
fi
echo ""

# Move analysis scripts
echo "Organizing analysis scripts..."
for file in analyze_affinity.sh analyze_affinity_distribution.py create_antibody_antigen_summary.sh extract_extreme_affinity.sh; do
    if [ -f "$file" ]; then
        mv "$file" scripts/analysis/
        echo "  ✅ Moved $file to scripts/analysis/"
    fi
done
echo ""

# Replace main README
echo "Updating main README..."
if [ -f "README_GITHUB.md" ]; then
    if [ -f "README.md" ]; then
        mv README.md archive/old_docs/README_original.md
        echo "  ✅ Backed up original README"
    fi
    cp README_GITHUB.md README.md
    echo "  ✅ README_GITHUB.md copied to README.md"
fi
echo ""

# Clean up redundant docs (prompt user)
echo "Redundant documentation files found:"
echo "  - docs/guides/QUICK_START.md"
echo "  - docs/guides/QUICK_START_GUIDE.md"
echo ""
echo "These overlap with QUICK_START_V2.md"
read -p "Move to archive? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv docs/guides/QUICK_START.md archive/old_docs/ 2>/dev/null
    mv docs/guides/QUICK_START_GUIDE.md archive/old_docs/ 2>/dev/null
    echo "  ✅ Moved redundant quick start guides to archive"
fi
echo ""

# Create .gitkeep files for empty important directories
touch models/.gitkeep
touch examples/.gitkeep
echo "✅ Created .gitkeep files"
echo ""

# Check for large files that shouldn't be committed
echo "Checking for large files..."
echo ""
echo "Files > 10MB (should be in .gitignore):"
find . -type f -size +10M ! -path "./.git/*" ! -path "./external_data/*" ! -path "./archive/*" 2>/dev/null | head -10
echo ""

# Summary
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo ""
echo "✅ Old versions moved to archive/old_versions/"
echo "✅ Old docs moved to archive/old_docs/"
echo "✅ Analysis scripts organized in scripts/analysis/"
echo "✅ Main README updated"
echo "✅ Directory structure created"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Test that code still works"
echo "3. Update any broken links in documentation"
echo "4. Run: git add ."
echo "5. Commit and push!"
echo ""
echo "=========================================="

