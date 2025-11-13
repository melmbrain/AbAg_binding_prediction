"""
Model Architecture for Full-Dimensional Training (v3)

This model uses 1,280-dimensional ESM2 embeddings (no PCA).
Optimized for Colab Pro with 16GB+ GPU memory.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class AffinityModelV3FullDim(nn.Module):
    """
    Full-dimensional affinity prediction model

    Architecture: 1,280 → 512 → 256 → 128 → 64 → 1

    Features:
    - GELU activation for smooth gradients
    - BatchNorm for stable training
    - Dropout for regularization
    - Xavier initialization
    """

    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dims: list = [512, 256, 128, 64],
                 dropout: float = 0.3,
                 use_batchnorm: bool = True):
        """
        Args:
            input_dim: Input feature dimension (1,280 for full ESM2)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate (0.3 recommended)
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)

            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # GELU activation (better than ReLU for deep networks)
            layers.append(nn.GELU())

            # Dropout
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, 1)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 1280)

        Returns:
            predictions: Output tensor of shape (batch_size,)
        """
        return self.network(x).squeeze(-1)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AffinityModelV3Deep(nn.Module):
    """
    Deeper architecture for full-dimensional features

    Architecture: 1,280 → 1,024 → 512 → 256 → 128 → 64 → 1

    Use this for maximum capacity if you have enough GPU memory.
    """

    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dims: list = [1024, 512, 256, 128, 64],
                 dropout: float = 0.35,
                 use_batchnorm: bool = True):
        """
        Args:
            input_dim: Input feature dimension (1,280 for full ESM2)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate (0.35 for deeper model)
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)

            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # GELU activation
            layers.append(nn.GELU())

            # Dropout (increase dropout in deeper layers)
            layer_dropout = dropout + (i * 0.05)  # Gradually increase
            layer_dropout = min(layer_dropout, 0.5)  # Cap at 0.5
            layers.append(nn.Dropout(layer_dropout))

            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, 1)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass"""
        return self.network(x).squeeze(-1)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AffinityModelV3WithAttention(nn.Module):
    """
    Full-dimensional model with attention mechanism

    The attention layer learns which features are most important
    for each sample, allowing the model to focus on relevant patterns.
    """

    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dims: list = [512, 256, 128, 64],
                 dropout: float = 0.3,
                 use_attention: bool = True):
        """
        Args:
            input_dim: Input feature dimension (1,280 for full ESM2)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        self.input_dim = input_dim
        self.use_attention = use_attention

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim),
                nn.Sigmoid()  # Attention weights between 0 and 1
            )

        # Main network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass with optional attention

        Args:
            x: Input tensor of shape (batch_size, 1280)

        Returns:
            predictions: Output tensor of shape (batch_size,)
        """
        if self.use_attention:
            # Apply attention weights to input
            attention_weights = self.attention(x)
            x = x * attention_weights  # Element-wise multiplication

        return self.network(x).squeeze(-1)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_v3(model_type='standard', **kwargs):
    """
    Factory function to get v3 models

    Args:
        model_type: 'standard', 'deep', or 'attention'
        **kwargs: Additional arguments passed to model constructor

    Returns:
        model: PyTorch model
    """
    if model_type == 'standard':
        model = AffinityModelV3FullDim(**kwargs)
    elif model_type == 'deep':
        model = AffinityModelV3Deep(**kwargs)
    elif model_type == 'attention':
        model = AffinityModelV3WithAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Created {model_type} model with {model.count_parameters():,} parameters")
    return model


# Example usage
if __name__ == "__main__":
    # Test standard model
    print("="*60)
    print("TESTING MODEL ARCHITECTURES")
    print("="*60)

    # Standard model
    print("\n1. Standard Model:")
    model_standard = get_model_v3('standard')
    x = torch.randn(32, 1280)  # Batch of 32 samples
    y = model_standard(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")

    # Deep model
    print("\n2. Deep Model:")
    model_deep = get_model_v3('deep')
    y = model_deep(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")

    # Attention model
    print("\n3. Attention Model:")
    model_attention = get_model_v3('attention')
    y = model_attention(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")

    print("\n" + "="*60)
    print("ALL MODELS WORKING CORRECTLY!")
    print("="*60)
