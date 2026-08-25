r"""
:mod:`cape.dkit.ae`: DataKit Autoencoder tools
======================================================

This module uses PyTorch to create and train Autoencoders.
"""

# Standard library
import copy
from typing import Dict, Optional, Tuple, Union

# Third party
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Local imports


# Encoder class
class Encoder(nn.Module):
    def __init__(
            self,
            input_length: int,
            latent_dim: int = 10,
            channel_list: Optional[list] = None,
            n: Optional[int] = None,
            stride: Union[list, int] = 2,
            kernel_size: Union[list, int] = 3,
            padding: Union[list, int] = 1,
            batchnorm: bool = True,
            dropout: float = 0.0,
            activation: Optional[nn.Module] = None):
        # Call parent function
        super().__init__()
        # Save input length
        self.input_length = input_length
        # Default list of layer output channels
        if channel_list is None:
            # Default number of layers
            if n is None:
                # Reduce by half until size is 1000
                n = max(4, int(np.log2(input_length / 1000)))
            # Number of channels for each layer: 16->8->4->1
            channel_list = np.flip(2**np.arange(n))
            channel_list[:-1] *= 2
        else:
            # Number of layers implied by list
            n = len(channel_list)
        # Convert inputs to list
        strides = _enlist(stride, n)
        kernel_sizes = _enlist(kernel_size, n)
        paddings = _enlist(padding, n)
        # Defaults
        activation = (
            nn.ReLU(inplace=True) if activation is None
            else activation)
        # Create actual layers
        blocks = []
        # Initialize number of input channels
        ch = 1
        # Loop through layers
        for j, out_ch in enumerate(channel_list):
            # Options for this layer
            sj = strides[j]
            fj = kernel_sizes[j]
            pj = paddings[j]
            # Initialize neural network layer
            layers = [nn.Conv1d(ch, out_ch, fj, sj, pj)]
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
        self.fc = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Call convolution layers
        feat = self.conv_blocks(x)
        # Call feature extraction layers
        z = self.fc(feat.flatten(1))
        # Output
        return feat, z


class ConvDecoder(nn.Module):
    def __init__(
            self,
            input_length: int,
            out_channels: int = 1,
            latent_dim: int = 10,
            channel_list: Optional[list] = None,
            n: Optional[int] = None,
            encoder_flat_size: int = 0,
            encoder_out_length: int = 50,
            stride: Union[list, int] = 2,
            kernel_size: Union[list, int] = 3,
            padding: Union[list, int] = 1,
            out_padding: Union[list, int] = 1,
            batchnorm: bool = True,
            dropout: float = 0.0,
            activation: Optional[nn.Module] = None,
            final_activation: Optional[nn.Module] = None):
        # Call parent function
        super().__init__()
        # Default list of layer output channels
        if channel_list is None:
            # Default number of layers
            n = 4 if n is None else n
            # Number of channels for each layer: 16->8->4->1
            channel_list = 2**np.arange(n)
            channel_list[1:] *= 2
        else:
            # Number of layers implied by list
            n = len(channel_list)
        # Convert inputs to list
        strides = _enlist(stride, n)
        kernel_sizes = _enlist(kernel_size, n)
        paddings = _enlist(padding, n)
        out_paddings = _enlist(out_padding, n)
        # Defaults
        activation = (
            nn.ReLU(inplace=True) if activation is None
            else activation)
        # Save parameters
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.encoder_out_length = encoder_out_length
        self.encoder_flat_size = encoder_flat_size
        self.first_ch = channel_list[0]
        # Set up linear decoder layer
        self.fc = nn.Linear(latent_dim, encoder_flat_size)
        # Set up inverse convolution blocks
        blocks = []
        for j in range(n - 1):
            # Channel size
            ch1 = 1
            ch2 = 1
            # Options for this layer
            sj = strides[j]
            fj = kernel_sizes[j]
            pj = paddings[j]
            oj = out_paddings[j]
            # Make decoder block
            layers = [
                nn.ConvTranspose1d(ch1, ch2, fj, sj, pj, oj)
            ]
            if batchnorm:
                layers.append(nn.BatchNorm1d(ch2))
            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            # Append block
            blocks.append(nn.Sequential(*layers))
        # Final layer: output channels and no batchnorm
        blocks.append(
            nn.Sequential(
                nn.ConvTranspose1d(
                    channel_list[-1], out_channels,
                    kernel_sizes[-1], strides[-1],
                    paddings[-1], out_paddings[-1])))
        # Save convolution blocks
        self.conv_blocks = nn.Sequential(*blocks)

    # Evaluation method
    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Fully connected layer
        x = self.fc(z)
        # Channels
        ch = self.encoder_flat_size // self.encoder_out_length
        x = x.view(x.size(0), ch, self.encoder_out_length)
        # Run convolution blocks
        return self.conv_blocks(x)[..., :self.input_length]


class ConvAutoencoder(nn.Module):
    r"""Full 1D convolution autoencoder

    :Call:
        >>> aec = ConvAutoencoder(**kw)
    :Inputs:
        *input_length*: {``400``} | :class:`int`
            Size of original vector
        *latent_dim*: {``10``} | :class:`int`
            Size or reduced vector
        *channel_list*: {``None``} | :class:`list`\ [:class:`int`]
            List of convolution channel numbers for each layer
        *n*: {``None``} | :class:`int`
            Number of convolution layers; matches ``len(channel_list)``
        *kernel_size*: {``3``} | :class:`int` | :class:`list`
            Convolution kernel size; can be different for each layer
        *stride*: {``1``} | :class:`int` | :class:`list`
            Stride parameter for each convolution layer
    """
    def __init__(
            self,
            input_length: int = 400,
            latent_dim: int = 10,
            channel_list: Optional[list] = None,
            n: Optional[int] = None,
            kernel_size: Union[int, list] = 3,
            stride: Union[int, list] = 2,
            padding: Union[int, list] = 1,
            out_padding: Union[int, list] = 1,
            activation: nn.Module = nn.LeakyReLU(0.2),
            final_activation: Optional[nn.Module] = None,
            batchnorm: bool = True,
            dropout: float = 0.2,
            weight_init: str = "kaiming"):
        # Parent initialization
        super().__init__()
        # Create encoder block
        self.encoder = Encoder(
            input_length, latent_dim, channel_list, n,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            batchnorm=batchnorm,
            dropout=dropout,
            activation=activation)
        # infer encoder conv output shape for decoder
        dummy = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            feat, _ = self.encoder(dummy)
        enc_out_length = feat.shape[2]
        enc_flat = feat.flatten(1).shape[1]
        # Create the decoder
        self.decoder = ConvDecoder(
            input_length,
            out_channels=1, latent_dim=latent_dim,
            channel_list=channel_list, n=n,
            encoder_flat_size=enc_flat, encoder_out_length=enc_out_length,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            out_padding=out_padding,
            batchnorm=batchnorm,
            dropout=dropout,
            activation=activation,
            final_activation=final_activation)
        # Initialize weights
        self._init_weights(weight_init)

    def _init_weights(self, mode: str):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if mode == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif mode == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                elif mode == "normal":
                    nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run encoder
        feat, z = self.encoder(x)
        # Run decoder
        recon = self.decoder(feat, z)
        return recon, z

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        return self(x)


class ReconstructionLoss(nn.Module):
    """Flexible reconstruction loss

    mode: `"mse"` | `"bce"` | `"mae"` | `"ssim"` | `"combined"`
        Error method
    For "combined", weights dict controls mixing, e.g.:
        weights={"mse": 0.8, "mae": 0.2}

    Note: SSIM is omitted for 1-D data (it is inherently a 2-D metric).
    """
    def __init__(
            self, mode: str = "mse",
            weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.mode    = mode
        self.weights = weights or {"mse": 1.0}
        self._losses = {
            "mse": nn.MSELoss(),
            "bce": nn.BCELoss(),
            "mae": nn.L1Loss(),
        }

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if self.mode == "combined":
            total = torch.tensor(0.0, device=pred.device)
            for name, w in self.weights.items():
                if name not in self._losses:
                    raise ValueError(
                        f"Unknown loss '{name}'; avail: {list(self._losses)}")
                total = total + w * self._losses[name](pred, target)
            return total
        if self.mode not in self._losses:
            raise ValueError(
                f"Unknown mode '{self.mode}'; avail: {list(self._losses)}")
        return self._losses[self.mode](pred, target)


def train_one_epoch(
        model: ConvAutoencoder,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler=None,
        clip_grad: float = 0.0) -> float:
    # Call autoencoder's train method
    model.train()
    # Count losses
    total_loss = 0.0
    for batch in loader:
        # Accept (x, label) tuples or raw tensors
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # Send to GPU if able
        x = x.to(device)
        # Optimizer methods
        optimizer.zero_grad()
        # Get prediction from current model
        recon, _ = model(x)
        # Evaluate loss
        loss = criterion(recon, x)
        # Calculate derivatives
        loss.backward()
        # Limit gradients
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        # Take the optimizer step
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    # Use scheduler if available
    if scheduler is not None:
        scheduler.step()
    # Normalize
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
        model: ConvAutoencoder,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        recon, _ = model(x)
        total_loss += criterion(recon, x).item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_model(
        model: ConvAutoencoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        device: torch.device,
        criterion: nn.Module = None,
        optimizer_cls=optim.Adam,
        optimizer_kwargs: dict = None,
        scheduler_cls=None,
        scheduler_kwargs: dict = None,
        clip_grad: float = 0.0,
        verbose: bool = True,
        return_best: bool = False) -> Dict[str, list]:
    criterion = criterion or ReconstructionLoss("mse")
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
    scheduler = scheduler_cls(
        optimizer, **(scheduler_kwargs or {})) if scheduler_cls else None
    history = {"train_loss": [], "val_loss": []}
    best_train_loss = 1e16
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, device, scheduler, clip_grad)
        history["train_loss"].append(train_loss)
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)

        # Save best model based on training loss
        if train_loss < best_train_loss and return_best:
            best_train_loss = train_loss
            # Deep copy the parameters
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            msg = f"Epoch [{epoch:>4}/{epochs}]  train={train_loss:.6f}"
            if val_loss is not None:
                msg += f"  val={val_loss:.6f}"
            print(msg)
    return history, best_state


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
