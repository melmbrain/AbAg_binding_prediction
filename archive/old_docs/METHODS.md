# Methods and Materials

## Overview
This document describes the methodology and materials used for antibody-antigen binding affinity prediction research.

---

## 1. Dataset

### 1.1 Data Sources
**Total Dataset: 390,757 antibody-antigen pairs**

| Database | Samples | Affinity Data | Very Strong (>11 pKd) | Description |
|----------|---------|---------------|----------------------|-------------|
| **AbBiBench** | 185,718 | Yes | - | Antibody binding benchmark dataset |
| **SAAINT-DB** | 6,158 | Yes | 173 | Structural antibody-antigen interaction database |
| **SAbDab** | 1,307 | Yes | 31 | Structural antibody database |
| **Phase 6** | 204,986 | Yes | 230 | Custom curated dataset |
| **Total** | **390,757** | **Yes** | **384 (0.1%)** | Integrated dataset |

- **Training samples (with complete features)**: 330,762
- **Affinity range**: pKd 0-16 (femtomolar to millimolar)
- **Format**: CSV files with sequences and affinity values
- **Primary affinity metric**: pKd (negative logarithm of dissociation constant Kd)

### 1.2 Data Structure
```
CSV columns:
- antibody_heavy: Heavy chain amino acid sequence
- antibody_light: Light chain amino acid sequence
- antigen: Antigen amino acid sequence
- pKd: Binding affinity (pKd scale, 0-16)
- Kd_nM: Dissociation constant in nanomolar (optional)
- database_source: Origin database (AbBiBench, SAAINT-DB, etc.)
- pdb_id: PDB identifier (if available)
```

### 1.3 Affinity Binning Strategy
Samples are categorized into 5 bins for stratified handling:

| Bin Label | pKd Range | Binding Strength | Typical Occurrence |
|-----------|-----------|------------------|-------------------|
| **very_weak** | 0-5 | Weak/No binding | ~15-20% |
| **weak** | 5-7 | Weak binding | ~20-25% |
| **moderate** | 7-9 | Moderate binding | ~40-50% |
| **strong** | 9-11 | Strong binding | ~10-15% |
| **very_strong** | >11 | Very strong binding | ~0.1% (rare) |

**Class Imbalance Challenge**: Very strong binders (>11 pKd) comprise only 0.1% of the dataset (384 samples), creating significant class imbalance requiring specialized handling.

### 1.4 Data Preprocessing
1. **Sequence Processing**
   - Filter samples with valid sequences for antibody heavy, light, and antigen
   - Remove entries with missing or invalid amino acid sequences
   - Standardize sequence format (uppercase, 20 standard amino acids)

2. **Affinity Value Processing**
   - Primary metric: pKd values
   - No outlier removal (all affinity ranges retained for diversity)
   - Distribution analysis per bin for stratification

3. **Train/Validation/Test Split**
   - **Train set**: 70% of data (~231,000 samples)
   - **Validation set**: 15% of data (~49,500 samples)
   - **Test set**: 15% of data (~49,500 samples)
   - **Stratification**: Maintains class distribution across all splits
   - **Random seed**: 42 (for reproducibility)
   - Split performed after filtering for complete features

---

## 2. Feature Extraction

### 2.1 Protein Language Model: ESM-2
- **Model**: ESM-2 (Evolutionary Scale Modeling v2)
- **Version**: `facebook/esm2_t33_650M_UR50D` from Hugging Face Transformers
- **Model Size**: 650M parameters, 33 transformer layers
- **Native Embedding Dimension**: 1,280 dimensions
- **Input**: Concatenated antibody-antigen sequences
- **Output**: Fixed-length embedding vectors per sequence

**Reference**: Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model" *Science*

### 2.2 Embedding Generation Process
1. **Load ESM-2 model and tokenizer**
   ```python
   from transformers import AutoTokenizer, AutoModel
   model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
   tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
   ```

2. **Sequence Processing**
   - Concatenate antibody heavy + light + antigen sequences
   - Add special tokens (BOS, EOS) as required by ESM-2
   - Tokenize using ESM-2 vocabulary (33 amino acid tokens)

3. **Extract Embeddings**
   - Forward pass through ESM-2 model
   - Extract representations from final layer (layer 33)
   - Apply mean pooling over sequence length dimension
   - Result: 1,280-dimensional vector per sample

4. **Feature Storage**
   - Save as CSV columns: `esm2_dim_0` to `esm2_dim_1279`
   - Alternative: Save as PyTorch tensors (`.pt` files) for faster loading

### 2.3 Feature Preprocessing

#### Version 2 (PCA-reduced features):
- **Original dimensions**: 1,280
- **Reduced dimensions**: 150 (via PCA)
- **Variance preserved**: 99.9%
- **Rationale**: Reduce computational cost and memory usage
- **Method**: sklearn.decomposition.PCA with 150 components
- **Feature columns**: `esm2_pca_0` to `esm2_pca_149`

#### Version 3 (Full-dimensional features):
- **Dimensions**: 1,280 (no reduction)
- **Variance preserved**: 100%
- **Rationale**: Maximum information retention for improved performance
- **Memory requirement**: 16GB+ GPU memory
- **Feature columns**: `esm2_dim_0` to `esm2_dim_1279`

#### Normalization:
- **Not applied** - ESM-2 embeddings are already well-scaled
- Model includes BatchNorm layers which handle normalization during training

---

## 3. Model Architecture

### 3.1 Evolution of Model Versions

#### Version 1 (Baseline)
```
Architecture: 150 → 512 → 256 → 128 → 64 → 1
Input: PCA-reduced features (150 dimensions)
Activation: ReLU
Normalization: BatchNorm1d
Dropout: 0.3
Initialization: Default PyTorch
Parameters: ~240,000
```

#### Version 2 (Improved - Current Production)
```
Architecture: 150 → 512 → 256 → 128 → 64 → 1
Input: PCA-reduced features (150 dimensions)
Activation: GELU (improved from ReLU)
Normalization: BatchNorm1d
Dropout: 0.3
Initialization: Xavier uniform
Parameters: ~240,000

Key Improvements over v1:
✓ GELU activation for smoother gradients
✓ Xavier initialization for better convergence
✓ Focal loss with 10x stronger class weights
✓ AdamW optimizer + Cosine annealing scheduler
```

**Performance (v2):**
- Overall RMSE: 1.38 (6.5% improvement over v1)
- MAE: 1.21
- Spearman ρ: 0.43
- Pearson r: 0.76
- R²: 0.58

#### Version 3 (Full-Dimensional)
```
Architecture: 1,280 → 512 → 256 → 128 → 64 → 1
Input: Full ESM-2 features (1,280 dimensions)
Activation: GELU
Normalization: BatchNorm1d
Dropout: 0.3
Initialization: Xavier uniform
Parameters: ~900,000

All v2 improvements + full dimensions (8.5x more features)
Expected: 10-30% improvement on extreme affinities
```

### 3.2 Detailed Layer Specifications

**Standard Model Architecture (v3):**

| Layer Type | Input Dim | Output Dim | Parameters |
|------------|-----------|------------|------------|
| Linear 1 | 1,280 | 512 | 655,872 |
| BatchNorm1d | 512 | 512 | 1,024 |
| GELU | 512 | 512 | 0 |
| Dropout (0.3) | 512 | 512 | 0 |
| Linear 2 | 512 | 256 | 131,328 |
| BatchNorm1d | 256 | 256 | 512 |
| GELU | 256 | 256 | 0 |
| Dropout (0.3) | 256 | 256 | 0 |
| Linear 3 | 256 | 128 | 32,896 |
| BatchNorm1d | 128 | 128 | 256 |
| GELU | 128 | 128 | 0 |
| Dropout (0.3) | 128 | 128 | 0 |
| Linear 4 | 128 | 64 | 8,256 |
| BatchNorm1d | 64 | 64 | 128 |
| GELU | 64 | 64 | 0 |
| Dropout (0.15) | 64 | 64 | 0 |
| Linear 5 (Output) | 64 | 1 | 65 |
| **Total** | | | **~900,000** |

**Activation Function Rationale:**
- **GELU (Gaussian Error Linear Unit)** chosen over ReLU
- Provides smoother gradients (differentiable everywhere)
- Better performance on regression tasks
- Formula: GELU(x) = x * Φ(x), where Φ is standard Gaussian CDF

**Dropout Strategy:**
- Standard dropout (0.3) in early layers
- Reduced dropout (0.15) in final hidden layer
- Prevents overfitting while maintaining capacity
- No dropout in output layer

### 3.3 Alternative Architectures (Available)

#### Deep Model (AffinityModelV3Deep)
```
Architecture: 1,280 → 1,024 → 512 → 256 → 128 → 64 → 1
Parameters: ~1.8M
Progressive dropout: 0.35 → 0.40 → 0.45 (capped at 0.5)
Use case: Maximum capacity for very large datasets
```

#### Attention Model (AffinityModelV3WithAttention)
```
Architecture: 1,280 → [Attention(320)] → 1,280 → 512 → 256 → 128 → 64 → 1
Attention mechanism: Self-attention to learn feature importance
Output: Weighted features + attention weights for interpretability
Use case: When feature importance analysis is needed
```

### 3.4 Model Initialization
- **Weight Initialization**: Xavier uniform (Glorot initialization)
  - Formula: U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
  - Better convergence for deep networks with GELU activation
- **Bias Initialization**: Zeros
- **BatchNorm**: Default PyTorch initialization (γ=1, β=0)

---

## 4. Training Procedure

### 4.1 Loss Functions

#### Focal MSE Loss (Primary - v2/v3)
Custom focal loss adapted for regression tasks to address class imbalance:

```python
FocalMSELoss(gamma=2.0, class_weights, bins)
```

**Formulation:**
1. Compute base MSE: `mse = (prediction - target)²`
2. Normalize error: `normalized_error = √mse / max_error`
3. Focal weight: `focal_weight = normalized_error^γ`
4. Apply class weights based on affinity bin
5. Final loss: `loss = mean(focal_weight * mse * class_weights)`

**Parameters:**
- `gamma=2.0`: Focusing parameter (higher values emphasize hard examples)
- `max_error=10.0`: Maximum expected error for normalization
- Class weights: 10x multiplier for extreme bins (very_weak, very_strong)

**Rationale:**
- Focuses learning on difficult-to-predict samples (large errors)
- Combines focal weighting with class-based weighting
- Addresses both sample difficulty and class imbalance

#### Weighted MSE Loss (Alternative)
Standard MSE with class-based weighting only:
```python
WeightedMSELoss(class_weights, bins)
```

**Class Weight Calculation:**
For each affinity bin:
1. Base weight: `w_base = total_samples / (n_bins × bin_count)`
2. Extreme multiplier: Apply 10x to very_weak and very_strong bins
3. Final weights typically range from 1.0 to 50.0

**Example weights (v2):**
- very_weak: 35.2 (10x emphasis)
- weak: 3.8
- moderate: 2.1
- strong: 8.5
- very_strong: 420.5 (10x emphasis, very rare)

### 4.2 Optimization

#### Optimizer: AdamW
```python
torch.optim.AdamW(
    lr=0.0001,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
```

**AdamW chosen over Adam:**
- Decoupled weight decay (better regularization)
- Improved generalization on deep networks
- Standard choice for transformer-based features

#### Learning Rate Schedule: Cosine Annealing with Warm Restarts
```python
CosineAnnealingWarmRestarts(
    T_0=20,           # Initial restart period
    T_mult=2,         # Period multiplier after restart
    eta_min=lr * 0.01 # Minimum learning rate (1e-6)
)
```

**Schedule behavior:**
- Starts at 1e-4, decreases to 1e-6 over 20 epochs
- Restarts at epoch 20 (to 1e-4)
- Next restart at epoch 40 (period doubled)
- Enables escape from local minima

**Alternative (v1):**
- ReduceLROnPlateau: Reduces LR by 0.5 when validation loss plateaus
- Patience: 5 epochs
- More conservative, fewer oscillations

### 4.3 Regularization Techniques

1. **Dropout**
   - Standard layers: 0.3 (30% neurons dropped)
   - Final hidden layer: 0.15 (reduced to retain capacity)
   - Applied during training only

2. **Weight Decay**
   - Value: 1e-4
   - Applied via AdamW optimizer
   - L2 penalty on all weights

3. **Gradient Clipping**
   - Max norm: 1.0
   - Prevents exploding gradients
   - Especially important with focal loss

4. **Batch Normalization**
   - Applied after each linear layer
   - Stabilizes training
   - Reduces internal covariate shift

5. **Early Stopping** (implicit)
   - Best model saved based on validation loss
   - Checkpoint saved when `val_loss < best_val_loss`
   - Prevents overfitting to training set

### 4.4 Training Configuration

**Version 2 (PCA features):**
- Batch size: 128
- Epochs: 100
- Training time: ~31 minutes (T4 GPU)
- GPU memory: ~6 GB
- Samples per epoch: ~231,000

**Version 3 (Full dimensions):**
- Batch size: 96 (reduced due to larger model)
- Epochs: 100
- Training time: ~12-15 hours (T4 GPU)
- GPU memory: ~14-15 GB
- Samples per epoch: ~231,000

**Hardware Requirements:**
| Version | GPU | VRAM | Colab Tier | Training Time |
|---------|-----|------|------------|---------------|
| v2 (PCA) | T4 | 6 GB | Free | 31 min |
| v3 (Full) | T4 | 14-15 GB | Pro | 12-15 hours |
| v3 (Full) | V100 | 14-15 GB | Pro+ | 6-8 hours |
| v3 (Full) | A100 | 14-15 GB | Pro+ | 3-4 hours |

### 4.5 Sampling Strategies

#### Stratified Batch Sampling (Primary)
```python
StratifiedBatchSampler(batch_size=96, shuffle=True)
```
- Ensures each batch contains proportional representation from all bins
- Maintains class distribution within batches
- Reduces gradient variance

#### Weighted Random Sampling (Alternative)
```python
WeightedRandomSampler(weights=class_weights, num_samples=len(dataset))
```
- Oversamples rare classes
- Higher probability of selecting very_strong samples
- Risk of overfitting to rare examples

#### Standard Sampling (Baseline)
- Random shuffling without stratification
- Used for comparison only

### 4.6 Data Augmentation
**Currently NOT implemented** - Potential future improvements:
- Sequence mutation (random AA substitutions)
- Noise injection to embeddings
- Synthetic sample generation using VAE/GAN

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

1. **Root Mean Square Error (RMSE)**
   ```python
   RMSE = √(Σ(y_pred - y_true)² / n)
   ```
   - **Unit**: pKd units (same as target)
   - **Interpretation**: Average magnitude of prediction error
   - **Lower is better** (0 = perfect predictions)
   - **v2 Performance**: 1.38 pKd units

2. **Mean Absolute Error (MAE)**
   ```python
   MAE = Σ|y_pred - y_true| / n
   ```
   - **Unit**: pKd units
   - **Interpretation**: Average absolute error, less sensitive to outliers than RMSE
   - **Lower is better**
   - **v2 Performance**: 1.21 pKd units

3. **Pearson Correlation Coefficient (r)**
   ```python
   r = cov(y_pred, y_true) / (σ_pred × σ_true)
   ```
   - **Range**: -1 to +1
   - **Interpretation**: Linear correlation strength
   - **Higher is better** (1 = perfect positive correlation)
   - **v2 Performance**: 0.76

4. **Coefficient of Determination (R²)**
   ```python
   R² = 1 - (SS_res / SS_tot)
   ```
   - **Range**: -∞ to 1 (typically 0 to 1)
   - **Interpretation**: Proportion of variance explained by model
   - **Higher is better** (1 = perfect fit)
   - **v2 Performance**: 0.58 (58% variance explained)

5. **Spearman's Rank Correlation (ρ)**
   ```python
   ρ = Pearson(rank(y_pred), rank(y_true))
   ```
   - **Range**: -1 to +1
   - **Interpretation**: Monotonic relationship strength (non-linear aware)
   - **Higher is better**
   - **v2 Performance**: 0.43
   - **Important for ranking applications** (e.g., candidate screening)

### 5.2 Per-Bin Metrics (Class-Specific Performance)

Metrics computed separately for each affinity bin to assess performance on specific binding strength ranges:

**v2 Performance Breakdown:**

| Bin | pKd Range | Test Samples | RMSE | MAE | Notes |
|-----|-----------|--------------|------|-----|-------|
| very_weak | 0-5 | ~12,000 | 0.85 | 0.72 | 24% improvement over v1 ✓ |
| weak | 5-7 | ~15,000 | 0.92 | 0.78 | Good performance |
| moderate | 7-9 | ~18,000 | 0.73 | 0.61 | 26% improvement over v1 ✓ |
| strong | 9-11 | ~4,000 | 1.18 | 0.95 | Acceptable |
| very_strong | >11 | ~50 | 2.53 | 2.01 | Challenging (only 50 test samples) ⚠️ |

**Key Observations:**
- Excellent performance on moderate affinities (7-9 pKd) - most common range
- Good performance on weak binders after class weighting improvements
- **Challenge**: Very strong binders (>11 pKd) remain difficult due to extreme rarity (0.1% of data)

### 5.3 Additional Diagnostic Metrics

1. **Residual Analysis**
   - Mean residual (should be ~0 for unbiased predictions)
   - Residual standard deviation
   - Residual distribution (check for normality)

2. **Prediction Distribution Analysis**
   - Compare std(predictions) vs std(targets)
   - Check for prediction collapse (underconfident model)
   - **v2 Observation**: Prediction std (1.66) < Target std (2.13) → model slightly underconfident on extremes

3. **Error by Affinity Range**
   - Systematic errors in specific ranges
   - Identifies where model struggles most

4. **Confidence Intervals** (future work)
   - Uncertainty estimation for predictions
   - Ensemble-based or Bayesian approaches

---

## 6. Validation Strategy

### 6.1 Hold-Out Validation (Primary Method)
**Three-way split approach:**
- **Training set**: 70% (~231,000 samples) - Model training only
- **Validation set**: 15% (~49,500 samples) - Hyperparameter tuning, early stopping
- **Test set**: 15% (~49,500 samples) - Final evaluation only (never used for tuning)

**Stratification:**
- All splits maintain affinity bin distribution
- Ensures balanced representation of rare classes in all sets
- Uses `train_test_split` with `stratify` parameter based on binned pKd values

**Random Seed:** 42 (fixed for reproducibility across all versions)

### 6.2 Validation Protocol

**During Training:**
1. Train on training set for 1 epoch
2. Evaluate on validation set
3. Save model if validation loss improves
4. Update learning rate scheduler based on validation loss
5. Repeat for 100 epochs

**Hyperparameter Selection:**
- All hyperparameters tuned using validation set performance
- Validation loss used as primary metric for model selection
- Per-bin RMSE monitored to ensure balanced performance

**Final Evaluation:**
- Load best model (lowest validation loss)
- Single evaluation on test set
- Report all metrics (RMSE, MAE, Pearson, Spearman, R²)
- **Test set never used for any training decisions**

### 6.3 Cross-Validation (Not Currently Used)
**Rationale for not using K-fold CV:**
- Dataset is large enough (390K samples) for reliable single split
- Training time is long (12-15 hours for v3)
- 5-fold CV would require 5× training time (60-75 hours)
- Hold-out validation provides sufficient confidence

**Potential future use:**
- For smaller subsets or specific analyses
- For ensemble model construction
- For uncertainty estimation

---

## 7. Experimental Setup

### 7.1 Computing Environment

**Primary Platform:** Google Colab Pro
- **GPU**: NVIDIA T4 (16GB VRAM)
- **RAM**: 12-25 GB (varies by session)
- **Storage**: Google Drive for data and model persistence
- **Cost**: $9.99/month for Colab Pro

**Alternative Platforms:**
- Google Colab Pro+ (V100/A100 GPUs for faster training)
- Local machines with NVIDIA GPUs (16GB+ VRAM)
- AWS/Azure cloud instances with GPU support

**Software Environment:**
- **OS**: Linux (Ubuntu 20.04 on Colab)
- **Python Version**: 3.10+
- **CUDA Version**: 11.8+ (for GPU acceleration)

### 7.2 Software Dependencies

**Core Deep Learning:**
```
torch>=1.12.0                 # PyTorch framework
transformers>=4.20.0          # Hugging Face (for ESM-2)
```

**Data Processing:**
```
numpy>=1.21.0                 # Numerical arrays
pandas>=1.3.0                 # DataFrames and CSV handling
scikit-learn>=1.0.0           # Metrics, PCA, splits
scipy>=1.7.0                  # Scientific computing (spearmanr)
```

**Utilities:**
```
tqdm>=4.62.0                  # Progress bars
matplotlib>=3.4.0             # Plotting (optional)
seaborn>=0.11.0               # Statistical plots (optional)
```

**Installation:**
```bash
pip install -r requirements.txt
```

### 7.3 Reproducibility

**Random Seeds (Fixed):**
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU
```

**Deterministic Operations:**
- Not explicitly enabled (for performance)
- Seeds ensure reproducible splits and initialization
- Minor variations possible due to GPU non-determinism

**Version Control:**
- Full codebase tracked in Git repository
- Model checkpoints saved with configuration
- Training logs and metrics saved for each run

**Reproducibility Checklist:**
- ✓ Fixed random seeds (42)
- ✓ Same train/val/test split across all versions
- ✓ Saved model configurations with checkpoints
- ✓ Documented hyperparameters
- ✓ Requirements.txt for exact library versions
- ✓ Training scripts available in repository

---

## 8. Results Storage

### 8.1 Model Checkpoints
- **Location**: `models/`
- **Naming Convention**: `best_model.pth`, `checkpoint_epoch_N.pth`
- **Content**: Model weights, optimizer state, training configuration

### 8.2 Training Logs
- **Location**: [Specify where logs are stored]
- **Content**:
  - Loss curves (training and validation)
  - Metric values per epoch
  - Learning rate schedule
  - Training time

### 8.3 Predictions
- **Format**: CSV files with columns [ID, True_Value, Predicted_Value]
- **Storage**: `results/` or designated output directory

---

## 9. Hyperparameter Tuning

### 9.1 Search Strategy
- **Method**: [Grid Search / Random Search / Bayesian Optimization]
- **Search Space**:
  ```
  - Learning rate: [1e-5, 1e-4, 1e-3]
  - Dropout: [0.2, 0.3, 0.4]
  - Hidden dimensions: [Various architectures]
  - Batch size: [16, 32, 64]
  ```

### 9.2 Selection Criteria
- **Primary Metric**: Validation loss
- **Secondary Considerations**: Training time, model size

---

## 10. Baseline Comparisons (if applicable)

### 10.1 Baseline Methods
- [Method 1]: [Brief description]
- [Method 2]: [Brief description]

### 10.2 Comparison Protocol
- Same train/test split
- Same evaluation metrics
- Fair comparison of computational resources

---

## 11. Limitations and Considerations

### 11.1 Known Limitations

**1. Class Imbalance (Critical Issue)**
- Very strong binders (>11 pKd): Only 384 samples (0.1% of dataset)
- Results in RMSE of 2.53 on very strong bin (vs 0.73-0.92 on other bins)
- Even 10x class weighting insufficient to fully overcome imbalance
- Model tends to predict toward mean (~7.5-8.0 pKd) for difficult cases

**2. Feature Compression (v2 only)**
- PCA reduction from 1,280 to 150 dimensions
- 99.9% variance preserved, but 0.1% may contain critical patterns
- Could limit performance on extreme affinities
- v3 addresses this with full 1,280 dimensions

**3. Sequence-Only Predictions**
- No 3D structural information used
- No binding site geometry or interface contacts
- No consideration of:
  - Post-translational modifications
  - Protein dynamics and flexibility
  - Solvent effects and pH conditions
  - Temperature effects

**4. Computational Requirements**
- v3 requires 16GB+ GPU (limits accessibility)
- 12-15 hour training time on T4 GPU
- ESM-2 embedding generation is time-consuming for new sequences

**5. Model Confidence**
- Model predictions have lower std (1.66) than true values (2.13)
- Indicates underconfidence, especially on extreme values
- No uncertainty estimates currently provided

### 11.2 Potential Biases

**Dataset Biases:**
- **Database bias**: Majority of data from AbBiBench (47% of samples)
- **Affinity range bias**: Heavy concentration in 7-9 pKd range (moderate binders)
- **Species bias**: Predominantly human and mouse antibodies
- **Antigen bias**: Some antigens over-represented in training data
- **Experimental condition bias**: Various labs, methods, conditions not standardized

**Model Biases:**
- **Regression to mean**: Tendency to predict values near mean affinity
- **Conservative predictions**: Avoids extreme predictions to minimize squared error
- **Sequence similarity bias**: May perform better on sequences similar to training data

**Evaluation Biases:**
- **Test set size**: Only ~50 very strong binders in test set (limited statistical power)
- **Bin boundary effects**: Samples near bin boundaries harder to classify
- **Metric selection**: RMSE penalizes extreme errors more than MAE

### 11.3 When NOT to Use This Model

**Inappropriate Use Cases:**
❌ High-precision prediction of sub-nanomolar affinities (>11 pKd)
❌ Production therapeutic development decisions without experimental validation
❌ Cases where very high accuracy is critical (e.g., regulatory submissions)
❌ Predicting affinities for non-standard antibody formats (e.g., nanobodies, bispecifics)
❌ Extrapolation to completely novel antibody or antigen classes

**Appropriate Use Cases:**
✓ Initial screening of moderate-affinity candidates (5-11 pKd)
✓ Ranking potential binders for prioritized experimental testing
✓ Filtering large antibody libraries (millions of candidates)
✓ Research and baseline comparisons
✓ Exploratory analysis and hypothesis generation

---

## 12. Future Directions

### 12.1 High Priority Improvements

**1. Full-Dimensional Training (v3 - In Progress)**
- Use complete 1,280-dimensional ESM-2 features (no PCA)
- Expected improvement: +10-30% on extreme affinities
- Addresses potential information loss from dimensionality reduction
- Status: Implementation complete, training in progress

**2. Two-Stage Training Strategy**
- **Stage 1**: Train on full balanced dataset (100 epochs)
- **Stage 2**: Fine-tune exclusively on extreme affinities (50 epochs)
- Forces model to specialize on difficult cases
- Expected improvement: +15-25% on very strong/weak binders
- Risk: Potential forgetting of moderate affinity patterns

**3. Ensemble Methods**
- Train 5 models with different random seeds
- Average predictions for improved robustness
- Provides uncertainty estimates (prediction variance)
- Expected improvement: +10-20% overall, especially on extremes
- Computational cost: 5× training time

### 12.2 Medium Priority Improvements

**4. Advanced Sampling Strategies**
- **Oversampling**: Duplicate rare class samples 10× per epoch
- **SMOTE for Embeddings**: Generate synthetic samples in embedding space
- **Curriculum Learning**: Gradually increase difficulty of examples
- Risk: Potential overfitting to rare classes

**5. Alternative Architectures**
- **Transformer-based**: Self-attention over sequence features
- **Graph Neural Networks**: Model antibody-antigen interaction graph
- **Residual Networks**: Deeper models with skip connections
- **Multi-task Learning**: Jointly predict pKd + binding category

**6. Uncertainty Quantification**
- **Monte Carlo Dropout**: Multiple forward passes for uncertainty
- **Ensemble Disagreement**: Variance across ensemble members
- **Bayesian Neural Networks**: Probabilistic weight distributions
- Enables confidence-aware predictions

### 12.3 Data Enhancement

**7. Additional Training Data**
- Target: 1,000+ very strong binders (currently 384)
- Sources:
  - Therapeutic antibody databases
  - Recent PDB releases
  - Synthetic data generation
  - Literature mining with affinity extraction

**8. Structural Information Integration**
- AlphaFold2 predicted structures
- Binding site identification
- Interface contact maps
- Geometric deep learning approaches

**9. Multi-modal Learning**
- Combine sequence AND structure features
- ESM-2 embeddings + structural embeddings
- Attention-based feature fusion

### 12.4 Long-term Research Directions

**10. Transfer Learning**
- Pre-train on related tasks:
  - Protein-protein interactions
  - Enzyme-substrate binding
  - General protein binding
- Fine-tune on antibody-antigen data

**11. Generative Models**
- VAE/GAN for synthetic training data
- Generate extreme affinity examples
- Data augmentation in feature space

**12. Explainability and Interpretability**
- Attention weight visualization
- Feature importance analysis (SHAP values)
- Identify critical residues for binding
- Guide experimental mutagenesis studies

---

## References

### Key Papers

1. **ESM-2 (Protein Language Model):**
   - Lin, Z., Akin, H., Rao, R., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637), 1123-1130.
   - DOI: 10.1126/science.ade2574

2. **Focal Loss (Adapted for Regression):**
   - Lin, T. Y., Goyal, P., Girshick, R., et al. (2017). "Focal loss for dense object detection." *Proceedings of the IEEE International Conference on Computer Vision*, 2980-2988.
   - Original paper for classification; adapted here for regression with class imbalance

3. **Class Imbalance in Machine Learning:**
   - He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

4. **AdamW Optimizer:**
   - Loshchilov, I., & Hutter, F. (2019). "Decoupled weight decay regularization." *International Conference on Learning Representations (ICLR)*.

### Datasets

1. **AbBiBench:**
   - Ecker, N., et al. (2024). "AbBiBench: A benchmark for antibody binding affinity prediction."
   - [Add DOI when available]

2. **SAAINT-DB (Structural Antibody-Antigen Interaction Database):**
   - Huang, S., et al. (2025). "SAAINT: A database of structurally annotated antibody-antigen interactions."
   - [Add DOI when available]

3. **SAbDab (Structural Antibody Database):**
   - Dunbar, J., Krawczyk, K., Leem, J., et al. (2014). "SAbDab: the structural antibody database." *Nucleic Acids Research*, 42(D1), D1140-D1146.
   - DOI: 10.1093/nar/gkt1043
   - URL: http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab

4. **Phase 6 Dataset:**
   - Custom curated dataset (internal)
   - 204,986 antibody-antigen pairs with binding affinities

### Software and Tools

**Deep Learning Framework:**
- PyTorch (v1.12+): Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." *NeurIPS*, 8026-8037.

**Transformers Library:**
- Hugging Face Transformers (v4.20+): Wolf, T., et al. (2020). "Transformers: State-of-the-art natural language processing." *EMNLP System Demonstrations*, 38-45.

**Scientific Computing:**
- NumPy: Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature*, 585(7825), 357-362.
- Pandas: McKinney, W. (2010). "Data structures for statistical computing in Python." *Proceedings of the 9th Python in Science Conference*.
- Scikit-learn: Pedregosa, F., et al. (2011). "Scikit-learn: Machine learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
- SciPy: Virtanen, P., et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods*, 17(3), 261-272.

---

## Appendix

### A. Hardware Specifications
- Detailed GPU specifications
- Memory requirements
- Training time estimates

### B. Hyperparameter Sensitivity Analysis
- Impact of learning rate
- Impact of model depth
- Impact of dropout rate

### C. Error Analysis
- Common failure cases
- Systematic errors
- Suggestions for improvement

---

**Document Version**: 1.0
**Last Updated**: [2025/11/06]
**Authors**: [Jaeseong Yoon]
**Contact**: [josh223@naver.com]
