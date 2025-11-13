# Experimental Outcomes & Future Research Plan

**Project**: Antibody-Antigen Binding Affinity Prediction
**Date**: November 2025
**Status**: Active Training

---

## 📊 Experimental Outcomes

### Experiment 1: ESM-2 Baseline (Incomplete)

**Configuration**:
- Model: Pure ESM-2 for both antibody and antigen
- Device: Local RTX 2060 GPU
- Training: 5/50 epochs completed

**Results**:
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Spearman | 0.4594 | 0.60-0.70 | ⚠️ Below target |
| Recall@pKd≥9 | 14.22% | 40-60% | ❌ Far below target |
| RMSE | 1.4467 | 1.25-1.35 | ⚠️ Above target |
| MAE | 1.3266 | - | - |
| Pearson | 0.7280 | - | - |

**Analysis**:
- ✅ Model is learning (Spearman improving from epoch 1)
- ❌ Severe underfitting (only 5/50 epochs)
- ❌ Low recall on strong binders (critical for drug discovery)
- ⚠️ Cannot draw conclusions - training incomplete

**Limitations**:
1. **Hardware**: RTX 2060 too slow (10s/batch = 36 days total)
2. **Early Stop**: CUDA error at epoch 6
3. **Insufficient Training**: 10% completion not enough for fair evaluation

**Conclusion**: **Inconclusive** - need full 50-epoch training to establish proper baseline

---

### Experiment 2: IgT5 + ESM-2 Hybrid (In Progress)

**Configuration**:
- Antibody Model: IgT5 (1024-dim, Dec 2024 state-of-the-art)
- Antigen Model: ESM-2 650M (1280-dim, proven on epitopes)
- Device: Google Colab T4 GPU
- Training: X/50 epochs (check Colab for current status)

**Expected Results** (based on literature):
| Metric | Expected Range | Basis |
|--------|---------------|-------|
| Spearman | 0.60-0.70 | IgT5 R² 0.297-0.306 on binding tasks |
| Recall@pKd≥9 | 40-60% | Antibody-specific features + focal loss |
| RMSE | 1.25-1.35 | Comparable to published benchmarks |

**Status**: Training in progress (started Nov 13, 2025)

**Actual Results**: *To be filled after training completes (~Nov 17-18, 2025)*

---

## 🎯 Performance Comparison (Planned)

### After Experiment 2 Completes

| Metric | ESM-2 (E5) | IgT5+ESM-2 (E50) | Improvement | Target Met? |
|--------|------------|------------------|-------------|-------------|
| Spearman | 0.46 | TBD | TBD | TBD |
| Recall@pKd≥9 | 14.22% | TBD | TBD | TBD |
| RMSE | 1.45 | TBD | TBD | TBD |

**Key Questions**:
1. Does IgT5 improve Recall@pKd≥9 significantly?
2. Is the +10-20% literature improvement reproducible?
3. Does antibody-specific model justify added complexity?

---

## 📈 Expected vs Actual (To Be Updated)

### Hypothesis
**H1**: Antibody-specific model (IgT5) will improve Recall@pKd≥9 by 15-30 percentage points
**H2**: Combined model will achieve Spearman ≥ 0.60
**H3**: Focal loss will emphasize strong binders effectively

### Results (Fill after training)
**H1**: ✅/❌ Recall improved from 14% to ___%
**H2**: ✅/❌ Spearman reached ___
**H3**: ✅/❌ Focal loss effect: ___

---

## 🔮 Future Research Plan

### Phase 1: Complete Current Training (Nov 13-18, 2025)
**Goal**: Establish IgT5 + ESM-2 as baseline

**Tasks**:
- [🔄] Monitor training progress daily
- [ ] Validate results on test set
- [ ] Analyze where model succeeds/fails
- [ ] Compare to literature benchmarks

**Deliverables**:
- Trained model (best_model.pth)
- Test set evaluation results
- Performance analysis report

---

### Phase 2: Baseline Evaluation (Nov 18-20, 2025)

**If Recall@pKd≥9 ≥ 40%** → SUCCESS, proceed to Phase 4

**If Recall@pKd≥9 = 30-40%** → Proceed to Phase 3

**If Recall@pKd≥9 < 30%** → Debug and investigate

---

### Phase 3: Optimization (If Needed)

#### Option A: Data-Level Improvements
**Rationale**: Address class imbalance for strong binders

**Approaches to Try**:
1. **Upsampling** strong binders (pKd ≥ 9)
   ```python
   strong_binders = df[df['pKd'] >= 9.0]
   df_balanced = pd.concat([df, strong_binders * 2])
   ```
   **Expected**: +5-10% Recall
   **Time**: 3-4 days

2. **Class weighting** in loss function
   ```python
   weights = torch.where(targets >= 9.0, 3.0, 1.0)
   loss = (criterion(pred, target) * weights).mean()
   ```
   **Expected**: +5-10% Recall
   **Time**: 3-4 days

3. **Hard negative mining**
   - Focus on samples where model fails
   **Expected**: +3-7% Recall
   **Time**: 4-5 days

#### Option B: Architecture Improvements
**Rationale**: Add structure-aware features

**Approaches to Try**:
1. **Add IgFold structure embeddings**
   ```python
   ab_bert = IgT5(ab_seq)  # 1024-dim
   ab_struct = IgFold.structure_embs(ab_seq)  # 64-dim
   ab_combined = concat([ab_bert, ab_struct])  # 1088-dim
   ```
   **Expected**: +5-15% Recall
   **Time**: 5-6 days

2. **Attention mechanism** between Ab-Ag embeddings
   ```python
   ab_emb = IgT5(ab_seq)
   ag_emb = ESM2(ag_seq)
   interaction = cross_attention(ab_emb, ag_emb)
   ```
   **Expected**: +10-20% Recall
   **Time**: 6-7 days

3. **Ensemble methods**
   - Combine IgT5+ESM-2, IgBERT+ESM-2, IgFold+ESM-2
   **Expected**: +5-10% Recall
   **Time**: 10 days (train 2 more models)

#### Option C: Loss Function Improvements
**Rationale**: Better emphasize extreme values

**Approaches to Try**:
1. **Quantile loss** (focus on high values)
   ```python
   criterion = QuantileLoss(quantile=0.9)
   ```
   **Expected**: +3-8% Recall
   **Time**: 3 days

2. **Multi-task learning** (predict binding + epitope)
   ```python
   loss = alpha * binding_loss + beta * epitope_loss
   ```
   **Expected**: +5-12% Recall
   **Time**: 5-6 days

3. **Ranking loss** (order samples correctly)
   ```python
   criterion = RankNetLoss()
   ```
   **Expected**: +2-5% Recall
   **Time**: 4 days

---

### Phase 4: Production Deployment (If Success)

**If Recall@pKd≥9 ≥ 40%:**

1. **Model Optimization**
   - Quantization (INT8) for faster inference
   - ONNX export for deployment
   - API wrapper for predictions

2. **Validation on External Data**
   - Test on independent antibody-antigen pairs
   - Validate on therapeutic antibodies
   - Compare to experimental binding assays

3. **Application Development**
   - Create prediction pipeline
   - Build confidence intervals
   - Develop visualization tools

---

### Phase 5: Advanced Research (Long-term)

#### Research Direction 1: Incorporate 3D Structure
**Rationale**: Binding is fundamentally a 3D interaction

**Approaches**:
1. Use AlphaFold3 for antibody-antigen complex prediction
2. Extract geometric features from predicted structures
3. Combine sequence + structure embeddings

**Expected Impact**: +15-25% Recall
**Timeline**: 2-3 months
**Resources**: High-compute GPUs for structure prediction

#### Research Direction 2: Few-Shot Learning
**Rationale**: Limited data for specific antibody types

**Approaches**:
1. Meta-learning across antibody families
2. Transfer learning from related tasks
3. Prototypical networks for rare binders

**Expected Impact**: Better generalization
**Timeline**: 1-2 months

#### Research Direction 3: Interpretability
**Rationale**: Understand why model predicts high/low binding

**Approaches**:
1. Attention visualization (which residues matter?)
2. Mutation effect prediction
3. CDR contribution analysis

**Expected Impact**: Trust and insights for drug design
**Timeline**: 2-4 weeks

#### Research Direction 4: Multi-Modal Learning
**Rationale**: Integrate multiple data types

**Approaches**:
1. Sequence + structure + biochemical properties
2. Evolutionary information (MSA)
3. Experimental features (pH, temperature)

**Expected Impact**: +10-20% overall performance
**Timeline**: 3-4 months

---

## 🎓 Research Gaps Identified

### Gap 1: Limited Benchmarks for Recall@pKd≥9
**Issue**: Most papers report Spearman/R², not recall on extreme values
**Impact**: Hard to set realistic targets
**Solution**: Create our own benchmark, publish results

### Gap 2: No Standard Train/Test Split
**Issue**: Different papers use different data splits
**Impact**: Hard to compare across papers
**Solution**: Use same split as SAbDab benchmark

### Gap 3: Antibody-Specific Antigen Models
**Issue**: All antigen models are general protein models
**Impact**: May miss epitope-specific patterns
**Solution**: Train epitope-specific model (future work)

### Gap 4: Paired Sequence-Structure Datasets
**Issue**: Limited data with both sequence and experimental structure
**Impact**: Can't fully leverage structure information
**Solution**: Use AlphaFold predictions as proxy

---

## 📊 Metrics for Future Experiments

### Primary Metrics
1. **Recall@pKd≥9**: % of strong binders correctly identified
2. **Precision@pKd≥9**: % of predicted strong binders that are correct
3. **F1@pKd≥9**: Harmonic mean of recall and precision

### Secondary Metrics
4. **Spearman correlation**: Overall ranking quality
5. **RMSE**: Overall prediction error
6. **MAE**: Average absolute error

### Specialized Metrics
7. **Recall@pKd≥10**: Ultra-strong binders (even more important)
8. **Top-K accuracy**: % of top 100 predicted pairs that are top 100 actual
9. **Stratified metrics**: Performance across different antibody types

---

## 🔬 Experimental Design for Phase 3

### Controlled Experiments
**Principle**: Change one thing at a time

**Template**:
```
Baseline: IgT5 + ESM-2 (current)
Variation 1: IgT5 + ESM-2 + Upsampling
Variation 2: IgT5 + ESM-2 + Class Weighting
...

Compare: Recall@pKd≥9 improvement
Control: Same data split, same hyperparameters
```

### Statistical Significance
**Goal**: Ensure improvements are real, not noise

**Method**:
1. Train each model 3 times with different random seeds
2. Report mean ± std dev
3. Use t-test to compare approaches

**Threshold**: p < 0.05 for significant difference

---

## 💡 Lessons to Apply in Future Work

### From Current Project
1. **Establish baseline first** before complex architectures
2. **Use cloud GPUs** from the start for large models
3. **Auto-detect model dimensions** - don't trust docs alone
4. **Save checkpoints frequently** - Colab can disconnect anytime
5. **Document decisions immediately** - hard to recreate later

### From Literature Review
1. **Latest ≠ best** - empirical validation needed
2. **Domain-specific helps** - antibody models beat general for antibody tasks
3. **Paired training matters** - heavy+light chain interaction is important
4. **Focal loss works** - confirmed across multiple papers
5. **Freeze encoders** - sufficient for most tasks

---

## 📝 Open Questions

### Scientific Questions
1. Why is Recall@pKd≥9 so low (14%) even with modern models?
2. Do CDR regions alone determine binding strength?
3. How much does sequence similarity to training data matter?
4. Can we predict binding without structure information?

### Technical Questions
1. What's the optimal batch size for IgT5 + ESM-2?
2. Does gradient accumulation help with small batches?
3. Which loss function best handles extreme values?
4. How many epochs are truly needed for convergence?

### Practical Questions
1. How to deploy 2.5GB model for production use?
2. What inference speed is acceptable for drug discovery?
3. How to update model as new antibodies are discovered?
4. Can we reduce model size without hurting performance?

---

## 🎯 Success Criteria (Refined)

### Minimum Viable Product
- ✅ Recall@pKd≥9 ≥ 30% (2x baseline)
- ✅ Spearman ≥ 0.55
- ✅ Inference time < 1s per sample

### Target Performance
- 🎯 Recall@pKd≥9 ≥ 40% (3x baseline)
- 🎯 Spearman ≥ 0.65
- 🎯 RMSE ≤ 1.30

### Stretch Goals
- ⭐ Recall@pKd≥9 ≥ 50% (3.5x baseline)
- ⭐ Spearman ≥ 0.70
- ⭐ Recall@pKd≥10 ≥ 30%

---

## 📅 Timeline

### Short-term (Nov 2025)
- **Nov 13-18**: Complete IgT5 + ESM-2 training
- **Nov 18-20**: Evaluate and analyze results
- **Nov 20-25**: Phase 3 optimizations (if needed)

### Medium-term (Dec 2025 - Jan 2026)
- **Dec**: Production deployment (if successful)
- **Jan**: External validation on independent data

### Long-term (2026)
- **Q1**: Advanced research (structure integration)
- **Q2**: Publication preparation
- **Q3**: Open-source release

---

## 🚀 Next Immediate Actions

### After Training Completes (Nov 17-18)

1. **Evaluate on Test Set**
   ```python
   test_metrics = evaluate(model, test_loader, device)
   print(f"Test Recall@pKd≥9: {test_metrics['recall_pkd9']:.2f}%")
   ```

2. **Analyze Predictions**
   - Where does model succeed?
   - Where does it fail?
   - Error distribution analysis

3. **Compare to Literature**
   - Our Spearman vs published benchmarks
   - Our Recall vs extrapolated values

4. **Make Go/No-Go Decision**
   - If Recall ≥ 40% → Deploy
   - If Recall = 30-40% → Optimize (Phase 3)
   - If Recall < 30% → Debug and investigate

---

## 📖 Publications Plan

### Target Venues
1. **Bioinformatics**: Computational method paper
2. **Nature Communications**: If results are exceptional
3. **PLOS Computational Biology**: Open-access option
4. **arXiv/bioRxiv**: Preprint first

### Paper Outline (Draft)
**Title**: "IgT5-ESM2: A Hybrid Protein Language Model for Antibody-Antigen Binding Affinity Prediction"

**Sections**:
1. Introduction
   - Drug discovery need for binding prediction
   - Limitations of current methods
2. Methods
   - IgT5 + ESM-2 architecture
   - Focal loss for extreme values
   - Training procedure
3. Results
   - Benchmark comparison
   - Ablation studies
   - Error analysis
4. Discussion
   - Why antibody-specific models help
   - Future directions
5. Conclusion

---

**Document Status**: Living document - update after each experiment
**Last Updated**: 2025-11-13
**Next Update**: After training completes (Nov 17-18, 2025)
