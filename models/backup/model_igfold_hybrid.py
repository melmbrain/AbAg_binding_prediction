"""
Hybrid IgFold + ESM-2 Model for Antibody-Antigen Binding Prediction

Architecture:
- Antibody: IgFold embeddings (512-dim BERT + 64-dim structure)
- Antigen: ESM-2 embeddings (1280-dim)
- Combines antibody-specific structural features with general protein features
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from igfold import IgFoldRunner
import numpy as np


class IgFoldESMHybridModel(nn.Module):
    """
    Hybrid model combining:
    - IgFold for antibody-specific features (structure + sequence)
    - ESM-2 for antigen features (sequence)
    """

    def __init__(
        self,
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        use_igfold_structure=True,
        use_igfold_bert=True,
        freeze_esm=True,
        dropout=0.3
    ):
        super().__init__()

        # IgFold for antibody features
        self.igfold = IgFoldRunner()
        self.use_igfold_structure = use_igfold_structure
        self.use_igfold_bert = use_igfold_bert

        # ESM-2 for antigen features
        self.esm = AutoModel.from_pretrained(esm_model_name)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

        # Calculate input dimensions
        ab_dim = 0
        if use_igfold_bert:
            ab_dim += 512  # BERT embeddings
        if use_igfold_structure:
            ab_dim += 64   # Structure embeddings

        esm_hidden = self.esm.config.hidden_size  # 1280 for t33_650M
        ag_dim = esm_hidden

        combined_dim = ab_dim + ag_dim

        # Attention mechanism for antibody embeddings
        if use_igfold_bert and use_igfold_structure:
            self.ab_attention = nn.Sequential(
                nn.Linear(ab_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        # Regressor head
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(1024),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

        print(f"\n{'='*70}")
        print(f"IgFold + ESM-2 Hybrid Model")
        print(f"{'='*70}")
        print(f"Antibody features: IgFold (BERT: {use_igfold_bert}, Structure: {use_igfold_structure})")
        print(f"  - Dimension: {ab_dim}")
        print(f"Antigen features: ESM-2 {esm_model_name}")
        print(f"  - Dimension: {ag_dim}")
        print(f"Combined dimension: {combined_dim}")
        print(f"{'='*70}\n")

    def get_antibody_embedding(self, antibody_seq):
        """
        Extract IgFold embeddings for antibody sequence

        Args:
            antibody_seq: str, antibody sequence (can be heavy+light or just heavy)

        Returns:
            torch.Tensor: pooled antibody embedding
        """
        # Parse sequence into heavy/light chains
        # Assume sequences are in format "HEAVY|||LIGHT" or just "HEAVY"
        if "|||" in antibody_seq:
            heavy, light = antibody_seq.split("|||")
            sequences = {"H": heavy, "L": light}
        else:
            # Just heavy chain
            sequences = {"H": antibody_seq}

        # Get IgFold embeddings
        with torch.no_grad():
            emb = self.igfold.embed(sequences=sequences)

        embeddings = []

        # Use BERT embeddings (512-dim)
        if self.use_igfold_bert:
            bert_emb = emb.bert_embs  # Shape: (1, L, 512)
            # Pool over sequence length (mean pooling)
            bert_pooled = bert_emb.mean(dim=1)  # (1, 512)
            embeddings.append(bert_pooled)

        # Use structure embeddings (64-dim)
        if self.use_igfold_structure:
            struct_emb = emb.structure_embs  # Shape: (1, L, 64)
            # Pool over sequence length (mean pooling)
            struct_pooled = struct_emb.mean(dim=1)  # (1, 64)
            embeddings.append(struct_pooled)

        # Concatenate embeddings
        ab_emb = torch.cat(embeddings, dim=-1)  # (1, 512+64)
        return ab_emb.squeeze(0)  # (512+64,)

    def get_antigen_embedding(self, antigen_seq, device):
        """
        Extract ESM-2 embeddings for antigen sequence

        Args:
            antigen_seq: str, antigen sequence
            device: torch device

        Returns:
            torch.Tensor: pooled antigen embedding
        """
        # Tokenize
        tokens = self.esm_tokenizer(
            antigen_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Get ESM-2 embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.esm(**tokens)
            ag_emb = outputs.last_hidden_state[:, 0, :]  # CLS token (1, 1280)

        return ag_emb.squeeze(0)  # (1280,)

    def forward(self, antibody_seqs, antigen_seqs, device):
        """
        Forward pass

        Args:
            antibody_seqs: List[str], batch of antibody sequences
            antigen_seqs: List[str], batch of antigen sequences
            device: torch device

        Returns:
            torch.Tensor: predicted pKd values (batch_size,)
        """
        batch_size = len(antibody_seqs)

        # Get antibody embeddings (IgFold)
        ab_embeddings = []
        for ab_seq in antibody_seqs:
            ab_emb = self.get_antibody_embedding(ab_seq)
            ab_embeddings.append(ab_emb)
        ab_embeddings = torch.stack(ab_embeddings).to(device)  # (batch_size, ab_dim)

        # Get antigen embeddings (ESM-2)
        ag_embeddings = []
        for ag_seq in antigen_seqs:
            ag_emb = self.get_antigen_embedding(ag_seq, device)
            ag_embeddings.append(ag_emb)
        ag_embeddings = torch.stack(ag_embeddings).to(device)  # (batch_size, ag_dim)

        # Combine embeddings
        combined = torch.cat([ab_embeddings, ag_embeddings], dim=1)

        # Predict pKd
        predictions = self.regressor(combined).squeeze(-1)

        return predictions


class FocalMSELoss(nn.Module):
    """Focal MSE Loss for emphasizing hard examples"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 + mse) ** self.gamma
        return (focal_weight * mse).mean()


# Simpler version without IgFold structure (faster training)
class IgFoldBERTOnly(IgFoldESMHybridModel):
    """
    Simplified version using only IgFold BERT embeddings (no structure prediction)
    Faster training, still antibody-specific
    """
    def __init__(self, **kwargs):
        kwargs['use_igfold_structure'] = False
        kwargs['use_igfold_bert'] = True
        super().__init__(**kwargs)


# Full version with all features
class IgFoldESMFull(IgFoldESMHybridModel):
    """
    Full version with both IgFold BERT + Structure embeddings
    Best performance, slower training
    """
    def __init__(self, **kwargs):
        kwargs['use_igfold_structure'] = True
        kwargs['use_igfold_bert'] = True
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing IgFold + ESM-2 Hybrid Model...")

    model = IgFoldESMFull()

    # Example sequences
    antibody_seqs = [
        "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVSS"
    ]
    antigen_seqs = [
        "MKTIIALSYIFCLVFADYKDDDDK"
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        predictions = model(antibody_seqs, antigen_seqs, device)
        print(f"\nPrediction: {predictions.item():.2f}")

    print("\nModel test successful!")
