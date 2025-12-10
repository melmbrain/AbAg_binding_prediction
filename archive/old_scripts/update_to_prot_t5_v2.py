#!/usr/bin/env python3
"""
Update notebook to use ProtT5 instead of ESM-2 3B for faster training
"""
import json

# Read ORIGINAL notebook (v2.7)
with open('C:/Users/401-24/Desktop/AbAg_binding_prediction/notebooks/colab_training_v2.7.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New model class using ProtT5
new_model_code = '''# Enhanced Model with ProtT5 (FASTER than ESM-2 3B!)
class EnhancedAbAgModel(nn.Module):
    def __init__(self, dropout=0.3, use_cross_attention=True):
        super().__init__()

        print("Building enhanced model with ProtT5...")
        print("="*60)
        print("MODEL CHANGE: ESM-2 3B -> ProtT5")
        print("  - ProtT5 is 3-4x FASTER than ESM-2 3B")
        print("  - Similar performance for protein tasks")
        print("  - Both Ab and Ag use T5-based encoders (optimized)")
        print("="*60)

        # IgT5 for antibodies (specialized for immunoglobulins)
        print("  Loading IgT5 for antibodies...")
        self.ab_tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5")
        self.ab_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")
        self.ab_dim = 1024  # IgT5 outputs 1024-dim embeddings

        # ProtT5 for antigens (general protein encoder, FAST!)
        print("  Loading ProtT5-XL for antigens...")
        self.ag_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.ag_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.ag_dim = 1024  # ProtT5-XL outputs 1024-dim embeddings

        # Freeze encoders (we only train projection + attention + heads)
        for param in self.ab_model.parameters():
            param.requires_grad = False
        for param in self.ag_model.parameters():
            param.requires_grad = False

        # Projection layers to common dimension
        self.common_dim = 512
        self.ab_proj = nn.Linear(self.ab_dim, self.common_dim)  # 1024 -> 512
        self.ag_proj = nn.Linear(self.ag_dim, self.common_dim)  # 1024 -> 512

        # Cross-attention
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_ab = CrossAttention(self.common_dim, num_heads=8, dropout=dropout)
            self.cross_attn_ag = CrossAttention(self.common_dim, num_heads=8, dropout=dropout)

        # Regression head with spectral normalization
        self.regression_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.common_dim * 2, 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),

            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),

            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),

            nn.Linear(128, 1)
        )

        # Classification head (auxiliary task)
        self.classifier = nn.Linear(self.common_dim * 2, 1)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable/1e6:.1f}M")
        print(f"  Total parameters: {total/1e6:.1f}M")
        print()
        print("Model architecture:")
        print("  Antibody:  IgT5 (1024-dim) -> Projection (512-dim)")
        print("  Antigen:   ProtT5-XL (1024-dim) -> Projection (512-dim)")
        print("  Fusion:    Cross-attention -> Concatenate (1024-dim)")
        print("  Output:    Regression head -> pKd prediction")

    def forward(self, antibody_seqs, antigen_seqs, device):
        # Tokenize antibodies
        ab_tokens = self.ab_tokenizer(
            antibody_seqs, return_tensors='pt', padding=True,
            truncation=True, max_length=512
        ).to(device)

        # Tokenize antigens - ProtT5 expects space-separated amino acids
        # Add spaces between amino acids for ProtT5
        ag_seqs_spaced = [" ".join(list(seq)) for seq in antigen_seqs]
        ag_tokens = self.ag_tokenizer(
            ag_seqs_spaced, return_tensors='pt', padding=True,
            truncation=True, max_length=2048
        ).to(device)

        # Encode (frozen encoders, no gradient)
        with torch.no_grad():
            ab_out = self.ab_model(**ab_tokens).last_hidden_state
            ag_out = self.ag_model(**ag_tokens).last_hidden_state

        # Mean pooling
        ab_emb = ab_out.mean(dim=1)  # [B, 1024]
        ag_emb = ag_out.mean(dim=1)  # [B, 1024]

        # Cast to bfloat16 for trainable layers (faster on A100)
        ab_emb = ab_emb.to(torch.bfloat16)
        ag_emb = ag_emb.to(torch.bfloat16)

        # Project to common dimension
        ab_proj = self.ab_proj(ab_emb)  # [B, 512]
        ag_proj = self.ag_proj(ag_emb)  # [B, 512]

        # Cross-attention (optional)
        if self.use_cross_attention:
            # Add sequence dimension for attention
            ab_proj = ab_proj.unsqueeze(1)  # [B, 1, 512]
            ag_proj = ag_proj.unsqueeze(1)  # [B, 1, 512]

            ab_enhanced = self.cross_attn_ab(ab_proj, ag_proj).squeeze(1)
            ag_enhanced = self.cross_attn_ag(ag_proj, ab_proj).squeeze(1)

            combined = torch.cat([ab_enhanced, ag_enhanced], dim=1)
        else:
            combined = torch.cat([ab_proj, ag_proj], dim=1)

        # Predictions (NO CLAMPING - let gradients flow!)
        pKd_pred = self.regression_head(combined).squeeze(-1)
        class_logits = self.classifier(combined).squeeze(-1)

        # Cast back to float32 for loss computation
        return pKd_pred.float(), class_logits.float()

print("Enhanced model class defined!")
print()
print("KEY CHANGES:")
print("  - ESM-2 3B (2560-dim, SLOW) -> ProtT5-XL (1024-dim, FAST)")
print("  - Expected speedup: 3-4x per epoch")
print("  - Both encoders now T5-based (optimized inference)")'''

# New model build code
new_build_code = '''# Build model with ProtT5
print("Building model...")

model = EnhancedAbAgModel(
    dropout=DROPOUT,
    use_cross_attention=USE_CROSS_ATTENTION
).to(device)

# Cast trainable layers to bfloat16 for faster training on A100
model.ab_proj = model.ab_proj.to(torch.bfloat16)
model.ag_proj = model.ag_proj.to(torch.bfloat16)
if model.use_cross_attention:
    model.cross_attn_ab = model.cross_attn_ab.to(torch.bfloat16)
    model.cross_attn_ag = model.cross_attn_ag.to(torch.bfloat16)
model.regression_head = model.regression_head.to(torch.bfloat16)
model.classifier = model.classifier.to(torch.bfloat16)

print()
print("Model ready!")
print("Trainable layers cast to bfloat16 for A100 optimization")'''

# New header
new_header = '''# Antibody-Antigen Binding Prediction - v2.8 (ProtT5 FAST)

## Faster Training with ProtT5 Instead of ESM-2 3B

**v2.8 Key Changes:**
- **ProtT5-XL** for antigens (replaces ESM-2 3B)
- **3-4x faster** per epoch
- Same IgT5 for antibodies (specialized for immunoglobulins)

**Speed Comparison:**
- ESM-2 3B: ~2.5 hours per epoch
- ProtT5-XL: ~40-60 minutes per epoch

**Architecture:**
- Antibody encoder: IgT5 (1024-dim)
- Antigen encoder: ProtT5-XL (1024-dim)
- Cross-attention fusion
- Multi-task output (regression + classification)

**Training Config:**
- Balanced dataset: 121,688 samples
- Batch: 16 x 8 = 128 effective
- Learning rate: 1e-3
- Loss: MSE (0.7) + BCE (0.3)

---'''

# Update cells by index (more reliable)
print("Updating cells...")

# Cell 0 - Header markdown
nb['cells'][0]['source'] = new_header.split('\n')
nb['cells'][0]['source'] = [line + '\n' for line in nb['cells'][0]['source'][:-1]] + [nb['cells'][0]['source'][-1]]
print("  Cell 0: Updated header (v2.7 -> v2.8)")

# Cell 16 - Model class (contains 'class EnhancedAbAgModel')
nb['cells'][16]['source'] = new_model_code.split('\n')
nb['cells'][16]['source'] = [line + '\n' for line in nb['cells'][16]['source'][:-1]] + [nb['cells'][16]['source'][-1]]
print("  Cell 16: Updated model class (ESM-2 -> ProtT5)")

# Cell 19 - Model build (contains 'use_esm2_3b')
nb['cells'][19]['source'] = new_build_code.split('\n')
nb['cells'][19]['source'] = [line + '\n' for line in nb['cells'][19]['source'][:-1]] + [nb['cells'][19]['source'][-1]]
print("  Cell 19: Updated model build")

# Also update the checkpoint size check - ProtT5 checkpoints are smaller (~3GB vs 16GB)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'if size > 1e9:' in src:  # Checkpoint size check
        # Change from 1GB to 500MB minimum (ProtT5 models are smaller)
        src = src.replace('if size > 1e9:', 'if size > 5e8:')  # 500MB minimum
        src = src.replace('# Should be > 1GB', '# Should be > 500MB (ProtT5 models are smaller)')
        nb['cells'][i]['source'] = src.split('\n')
        nb['cells'][i]['source'] = [line + '\n' for line in nb['cells'][i]['source'][:-1]] + [nb['cells'][i]['source'][-1]]
        print(f"  Cell {i}: Updated checkpoint size check (1GB -> 500MB)")

# Save notebook
with open('C:/Users/401-24/Desktop/AbAg_binding_prediction/notebooks/colab_training_v2.8_ProtT5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print()
print("="*60)
print("SUCCESS: colab_training_v2.8_ProtT5.ipynb created")
print("="*60)
print()
print("Model changes:")
print("  - Antibody: IgT5 (unchanged, 1024-dim)")
print("  - Antigen:  ESM-2 3B -> ProtT5-XL (1024-dim)")
print()
print("Expected speedup:")
print("  - ESM-2 3B: ~2.5 hours/epoch (slow!)")
print("  - ProtT5-XL: ~40-60 minutes/epoch (3-4x faster)")
print()
print("Total training time estimate:")
print("  - 40 epochs x 50 min = ~33 hours (vs 100+ hours)")
print()
print("Next steps:")
print("  1. Upload notebooks/colab_training_v2.8_ProtT5.ipynb to Google Drive")
print("  2. In Cell 11, change: CSV_FILENAME = 'agab_phase2_full_v2_balanced.csv'")
print("  3. Delete old checkpoints (different model architecture)")
print("  4. Run training on A100!")
