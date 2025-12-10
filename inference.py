"""
Antibody-Antigen Binding Affinity Prediction - Inference Script
================================================================

This script predicts binding affinity (pKd) for antibody-antigen pairs
using the trained Stage 2 model.

Usage:
    python inference.py --antibody "EVQLVESGGGLVQPGG..." --antigen "MKTIIALSYIF..."
    python inference.py --csv input.csv --output predictions.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================================
# Model Architecture (must match training)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, ab_dim, ag_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.ab_proj = nn.Linear(ab_dim, hidden_dim)
        self.ag_proj = nn.Linear(ag_dim, hidden_dim)
        self.cross_attn_ab = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.cross_attn_ag = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output_dim = hidden_dim

    def forward(self, ab, ag):
        ab_p = self.ab_proj(ab).unsqueeze(1)
        ag_p = self.ag_proj(ag).unsqueeze(1)
        ab_att, _ = self.cross_attn_ab(ab_p, ag_p, ag_p)
        ag_att, _ = self.cross_attn_ag(ag_p, ab_p, ab_p)
        return self.combine(torch.cat([ab_att.squeeze(1), ag_att.squeeze(1)], dim=-1))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2, use_bn=True):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList()
        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h),
                nn.BatchNorm1d(h) if use_bn else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            prev = h
        self.output = nn.Linear(prev, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class AffinityPredictor(nn.Module):
    def __init__(self, ab_dim, ag_dim, config):
        super().__init__()
        hidden_dim = config.get('fusion_hidden_dim', 512)
        num_heads = config.get('fusion_num_heads', 8)
        dropout = config.get('fusion_dropout', 0.1)
        head_dims = config.get('head_hidden_dims', [512, 256, 128])
        head_dropout = config.get('head_dropout', 0.2)

        self.fusion = CrossAttentionFusion(ab_dim, ag_dim, hidden_dim, num_heads, dropout)
        self.head = ResidualMLP(self.fusion.output_dim, head_dims, head_dropout, use_bn=True)

    def forward(self, ab, ag):
        return self.head(self.fusion(ab, ag))


# ============================================================================
# Predictor Class
# ============================================================================

class BindingAffinityPredictor:
    """Predict antibody-antigen binding affinity."""

    def __init__(self, model_path, embeddings_dir=None, device=None):
        """
        Initialize predictor.

        Args:
            model_path: Path to the trained .pth model file
            embeddings_dir: Directory containing pre-computed embeddings (optional)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.ab_dim = checkpoint['ab_dim']
        self.ag_dim = checkpoint['ag_dim']

        self.model = AffinityPredictor(self.ab_dim, self.ag_dim, self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded: R²={checkpoint['val_metrics']['r2']:.4f}")

        # Load encoders for sequence-based prediction
        self.encoders_loaded = False
        self.ab_model = None
        self.ag_model = None
        self.ab_tokenizer = None
        self.ag_tokenizer = None

    def load_encoders(self):
        """Load protein language models for sequence encoding."""
        if self.encoders_loaded:
            return

        print("Loading protein encoders (this may take a minute)...")
        from transformers import T5EncoderModel, T5Tokenizer

        # Antibody encoder (IgT5)
        self.ab_tokenizer = T5Tokenizer.from_pretrained('Exscientia/IgT5')
        self.ab_model = T5EncoderModel.from_pretrained(
            'Exscientia/IgT5',
            torch_dtype=torch.float16
        ).to(self.device)
        self.ab_model.eval()

        # Antigen encoder (ProtT5)
        self.ag_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.ag_model = T5EncoderModel.from_pretrained(
            'Rostlab/prot_t5_xl_half_uniref50-enc',
            torch_dtype=torch.float16
        ).to(self.device)
        self.ag_model.eval()

        self.encoders_loaded = True
        print("Encoders loaded!")

    def encode_sequence(self, sequence, is_antibody=True):
        """Encode a protein sequence to embedding."""
        self.load_encoders()

        # Prepare sequence (space-separated amino acids)
        seq = " ".join(list(sequence.upper().replace(" ", "")))

        if is_antibody:
            tokenizer = self.ab_tokenizer
            model = self.ab_model
            max_length = 256
        else:
            tokenizer = self.ag_tokenizer
            model = self.ag_model
            max_length = 512

        tokens = tokenizer(seq, return_tensors='pt', padding=True,
                          truncation=True, max_length=max_length).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(**tokens)
            hidden = output.last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            embedding = (hidden * mask).sum(1) / mask.sum(1)

        return embedding.float()

    def predict_from_sequences(self, antibody_seq, antigen_seq):
        """
        Predict binding affinity from sequences.

        Args:
            antibody_seq: Antibody amino acid sequence (str)
            antigen_seq: Antigen amino acid sequence (str)

        Returns:
            pKd prediction (float)
        """
        ab_emb = self.encode_sequence(antibody_seq, is_antibody=True)
        ag_emb = self.encode_sequence(antigen_seq, is_antibody=False)

        with torch.no_grad():
            pred = self.model(ab_emb, ag_emb)

        return pred.item()

    def predict_from_embeddings(self, ab_embedding, ag_embedding):
        """
        Predict binding affinity from pre-computed embeddings.

        Args:
            ab_embedding: Antibody embedding (numpy array or tensor)
            ag_embedding: Antigen embedding (numpy array or tensor)

        Returns:
            pKd prediction (float or array)
        """
        if isinstance(ab_embedding, np.ndarray):
            ab_embedding = torch.tensor(ab_embedding, dtype=torch.float32)
        if isinstance(ag_embedding, np.ndarray):
            ag_embedding = torch.tensor(ag_embedding, dtype=torch.float32)

        ab_embedding = ab_embedding.to(self.device)
        ag_embedding = ag_embedding.to(self.device)

        # Handle batch dimension
        if ab_embedding.dim() == 1:
            ab_embedding = ab_embedding.unsqueeze(0)
            ag_embedding = ag_embedding.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(ab_embedding, ag_embedding)

        return pred.cpu().numpy()

    def predict_batch(self, antibody_seqs, antigen_seqs, batch_size=8):
        """
        Predict binding affinity for multiple pairs.

        Args:
            antibody_seqs: List of antibody sequences
            antigen_seqs: List of antigen sequences
            batch_size: Batch size for processing

        Returns:
            List of pKd predictions
        """
        self.load_encoders()
        predictions = []

        for i in range(0, len(antibody_seqs), batch_size):
            batch_ab = antibody_seqs[i:i+batch_size]
            batch_ag = antigen_seqs[i:i+batch_size]

            # Encode batch
            ab_seqs = [" ".join(list(s.upper().replace(" ", ""))) for s in batch_ab]
            ag_seqs = [" ".join(list(s.upper().replace(" ", ""))) for s in batch_ag]

            ab_tokens = self.ab_tokenizer(ab_seqs, return_tensors='pt', padding=True,
                                          truncation=True, max_length=256).to(self.device)
            ag_tokens = self.ag_tokenizer(ag_seqs, return_tensors='pt', padding=True,
                                          truncation=True, max_length=512).to(self.device)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                ab_out = self.ab_model(**ab_tokens)
                ag_out = self.ag_model(**ag_tokens)

                mask_ab = ab_tokens['attention_mask'].unsqueeze(-1).float()
                mask_ag = ag_tokens['attention_mask'].unsqueeze(-1).float()
                ab_emb = (ab_out.last_hidden_state * mask_ab).sum(1) / mask_ab.sum(1)
                ag_emb = (ag_out.last_hidden_state * mask_ag).sum(1) / mask_ag.sum(1)

                preds = self.model(ab_emb.float(), ag_emb.float())

            predictions.extend(preds.cpu().numpy().tolist())

        return predictions

    def predict_csv(self, input_csv, output_csv=None, ab_col='antibody_sequence',
                    ag_col='antigen_sequence', batch_size=8):
        """
        Predict binding affinity for a CSV file.

        Args:
            input_csv: Path to input CSV with antibody and antigen sequences
            output_csv: Path to save predictions (default: input_predictions.csv)
            ab_col: Column name for antibody sequences
            ag_col: Column name for antigen sequences
            batch_size: Batch size for processing

        Returns:
            DataFrame with predictions
        """
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} samples from {input_csv}")

        predictions = self.predict_batch(
            df[ab_col].tolist(),
            df[ag_col].tolist(),
            batch_size=batch_size
        )

        df['predicted_pKd'] = predictions
        df['predicted_Kd_nM'] = 10 ** (9 - df['predicted_pKd'])  # Convert to nM

        if output_csv is None:
            output_csv = input_csv.replace('.csv', '_predictions.csv')

        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

        return df


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predict antibody-antigen binding affinity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python inference.py --antibody "EVQLVESGGGLVQPGG..." --antigen "MKTIIALSYIF..."

  # Batch prediction from CSV
  python inference.py --csv input.csv --output predictions.csv

  # Custom model path
  python inference.py --model path/to/model.pth --csv input.csv
        """
    )

    parser.add_argument('--model', type=str,
                       default='models/stage2_final.pth',
                       help='Path to trained model file')
    parser.add_argument('--antibody', type=str, help='Antibody sequence')
    parser.add_argument('--antigen', type=str, help='Antigen sequence')
    parser.add_argument('--csv', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--ab-col', type=str, default='antibody_sequence',
                       help='Antibody column name in CSV')
    parser.add_argument('--ag-col', type=str, default='antigen_sequence',
                       help='Antigen column name in CSV')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Initialize predictor
    predictor = BindingAffinityPredictor(args.model, device=args.device)

    if args.csv:
        # Batch prediction
        df = predictor.predict_csv(
            args.csv,
            args.output,
            ab_col=args.ab_col,
            ag_col=args.ag_col,
            batch_size=args.batch_size
        )
        print(f"\nPrediction statistics:")
        print(f"  Mean pKd: {df['predicted_pKd'].mean():.2f}")
        print(f"  Std pKd:  {df['predicted_pKd'].std():.2f}")
        print(f"  Min pKd:  {df['predicted_pKd'].min():.2f}")
        print(f"  Max pKd:  {df['predicted_pKd'].max():.2f}")

    elif args.antibody and args.antigen:
        # Single prediction
        pKd = predictor.predict_from_sequences(args.antibody, args.antigen)
        Kd_nM = 10 ** (9 - pKd)

        print(f"\n{'='*50}")
        print("PREDICTION RESULT")
        print('='*50)
        print(f"  pKd:  {pKd:.2f}")
        print(f"  Kd:   {Kd_nM:.2f} nM")
        print(f"  Kd:   {Kd_nM/1000:.4f} µM")

        # Interpret result
        if pKd >= 9:
            print(f"  Binding: Very Strong (pKd >= 9)")
        elif pKd >= 7:
            print(f"  Binding: Strong (7 <= pKd < 9)")
        elif pKd >= 5:
            print(f"  Binding: Moderate (5 <= pKd < 7)")
        else:
            print(f"  Binding: Weak (pKd < 5)")
        print('='*50)
    else:
        parser.print_help()
        print("\nError: Provide either --antibody and --antigen, or --csv")


if __name__ == '__main__':
    main()
