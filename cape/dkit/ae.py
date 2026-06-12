r"""
:mod:`cape.dkit.ae`: DataKit Autoencoder tools
======================================================

This module uses PyTorch to create and train Autoencoders.
"""

# Standard library
import copy
from typing import Optional, Tuple, Union

# Third party
import numpy as np
import torch
from torch import nn

# Local imports
from .rdb import DataKit


# Encoder class
class Encoder(nn.Module):
    def __init__(
            self,
            input_length: int,
            n: Optional[int] = None,
            stride: Union[list, int] = 2,
            batchnorm: bool = True,
            dropout: float = 0.0,
            activation: Optional[nn.Module] = None
    ):
        # Call parent function
        super().__init__()
        # Default number of layers
        if n is None:
            # Reduce by half until size is 1000
            n = max(4, int(np.log2(input_length / 1000)))
        # Convert inputs to list
        strides = _enlist(stride, n)
        # Defaults
        activation = (
            nn.ReLU(inplace=True) if activation is None
            else activation)
        # Number of channels for each layer
        channel_list = np.flip(2**np.arange(n))
        # Create actual layers
        blocks = []
        # Initialize number of input channels
        ch = 1
        # Loop through layers
        for j, out_ch in enumerate(channel_list):
            # Options for this layer
            sj = strides[j]
            # Initialize neural network layer
            layers = [nn.Conv1d(ch, out_ch, 3, sj, 1)]
            # Append batch norm option if implied
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            # Add activation layer
            layers.append(copy.deepcopy(activation))
            # Add dropout layer
            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            # Convert to NN
            blocks.append(nn.Sequential(*layers))
            # Update previous layer size
            ch = out_ch
        # Save overall encoder
        self.conv_blocks = nn.Sequential(*blocks)
        # Dimension of output
        self.latent_dim = 1
        # Infer flattened size after all conv blocks
        dummy = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            flat = self.conv_blocks(dummy).flatten(1).shape[1]
        self.flat_size = flat
        # Create callable feature extraction layer
        self.fc = nn.Linear(flat, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Call convolution layers
        feat = self.conv_blocks(x)           # (B, C, L')
        # Call feature extraction layers
        z = self.fc(feat.flatten(1))
        # Output
        return feat, z


def _enlist(v: Union[list, int], n: int) -> list:
    # Check input
    if isinstance(v, (list, tuple)):
        # Check size
        if len(v) != n:
            raise IndexError(f"Expected size {n}; got {len(v)}")
        # Good
        return list(v)
    else:
        # Convert to list of size *n*
        return [v] * n
