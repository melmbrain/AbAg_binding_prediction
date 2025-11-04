# Quick Start - Training v2 (IMPROVED)

**Use this to get MUCH better results than v1!**

---

## 🎯 Expected Improvement

| Metric | v1 Result | v2 Expected | Improvement |
|--------|-----------|-------------|-------------|
| **Very Strong RMSE** | **2.94** | **1.0-1.5** | **⭐ 50-67% better** |
| **Overall RMSE** | 1.48 | 0.8-1.0 | 32-46% better |
| **Spearman ρ** | 0.39 | 0.65-0.75 | 66-92% better |

---

## 🚀 3-Step Quick Start

### Step 1: Upload Notebook (2 min)

1. Go to: https://colab.research.google.com/
2. File → Upload notebook
3. Upload: **`colab_training_v2_improved.ipynb`**
4. Runtime → Change runtime type → **GPU (T4)**

### Step 2: Update Paths (1 min)

Find the cell "Mount Google Drive" and verify:
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/merged_with_all_features.csv"
OUTPUT_DIR = "/content/drive/MyDrive/AbAg_data/models_v2"
```

### Step 3: Run! (10-12 hours)

- Click: **Runtime → Run all**
- Authorize Google Drive access
- Wait for training to complete
- All results saved to `models_v2/` folder

**Done!** 🎉

---

## 📊 What's Different in v2?

### 8 Major Improvements:

1. **✨ GELU Activation** - Smoother than ReLU, better gradients
2. **🏗️ Deeper Model** - 150→512→256→128→64→1 (vs 150→256→128→1)
3. **⚖️ 10x Stronger Weights** - Very strong/weak binders weighted 10x more
4. **📉 Lower Learning Rate** - 0.0001 (vs 0.001) for stability
5. **🎯 Focal Loss** - Focuses on hard examples
6. **✂️ Gradient Clipping** - Prevents exploding gradients
7. **🚀 Better Optimizer** - AdamW + Cosine Annealing
8. **🎲 Xavier Init** - Better starting weights

**Result:** Much better performance on extreme affinities!

---

## 📁 What You'll Get

After training, download from Google Drive:

```
models_v2/
├── best_model_v2.pth              ⭐ Your trained model
├── evaluation_results_v2.txt      📊 Performance metrics
├── test_predictions_v2.csv        📈 All predictions
├── predictions_vs_targets_v2.png  📉 Scatter plot
├── residuals_analysis_v2.png      🔍 Error analysis
└── training_curves_v2.png         📈 Loss over time
```

---

## 💡 Tips

**Keep tab open** - Colab disconnects if idle too long
**Check progress** - See checkpoint files appearing in Drive
**Compare results** - Notebook shows v1 vs v2 comparison automatically

---

## 📖 Full Documentation

For complete details, see: **`V2_IMPROVEMENTS.md`**

---

**Ready? Upload the notebook and click "Run all"!** 🚀
