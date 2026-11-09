import torch
import torch.nn as nn


class CNN3DBlock(nn.Module):
    """3D CNN block with ReLU activation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int]):
        super().__init__()
        # If you want to keep the exact shape progression from Table 1, note that
        # kernel_size=(7,3,3) in PyTorch means a 7-band "depth" kernel,
        # while the next two 3,3 are the spatial dims.
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class CNN2DBlock(nn.Module):
    """2D CNN block with ReLU activation"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class HSSNB(nn.Module):
    """
    Hybrid Spectral-Spatial Network with Bi-LSTM (HSSNB)
    
    Architecture from:
    "Hybrid CNN Bi-LSTM neural network for Hyperspectral image classification"
    Alok Ranjan Sahoo and Pavan Chakraborty
    """
    def __init__(self, num_classes: int = 16, n_components: int = 30):
        super().__init__()
        # The three 3D kernels shrink the spectral axis by 6 + 4 + 2 bands.
        spectral_out = n_components - 12
        if spectral_out < 1:
            raise ValueError(f"n_components must be at least 13, got {n_components}")

        # 3D CNN path 
        # (paper says kernel sizes: 3x3x7, 3x3x5, 3x3x3, but in code we keep
        #  (7,3,3), (5,3,3), (3,3,3) to match PyTorch ordering of (D,H,W))
        self.conv3d_1 = CNN3DBlock(1, 8,  kernel_size=(7,3,3))
        self.conv3d_2 = CNN3DBlock(8, 16, kernel_size=(5,3,3))
        self.conv3d_3 = CNN3DBlock(16, 32, kernel_size=(3,3,3))
        
        # 2D CNN path
        # After the 3D CNN, we flatten [channel x spectral] -> new "channel"
        self.conv2d_1 = CNN2DBlock(32 * spectral_out, 64)   # 32 * 18 == 576 by default
        self.conv2d_2 = CNN2DBlock(64, 128)
        
        # Optional: A layer norm before Bi-LSTM can help stabilize training
        self.layer_norm = nn.LayerNorm(128)
        
        # Bi-LSTM path
        self.bilstm_1 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.25)
        self.bilstm_2 = nn.LSTM(
            input_size=128,  # 64 * 2 for bidirectional
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Input shape: (batch_size, 1, n_components, window, window), by default
        (batch_size, 1, 30, 25, 25). The shape comments below are for that default.
        The window must be at least 11 pixels, since the five convolutions each
        trim 2 pixels from the spatial size.
        """
        batch_size = x.size(0)
        
        # 3D CNN (spectral + spatial)
        x = self.conv3d_1(x)  # -> (batch, 8,  24, 23, 23)
        x = self.conv3d_2(x)  # -> (batch, 16, 20, 21, 21)
        x = self.conv3d_3(x)  # -> (batch, 32, 18, 19, 19)
        
        # Reshape: combine "channel=32" and "spectral=18" => 576
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # -> (batch, 576, 19, 19)
        
        # 2D CNN (spatial feature extraction)
        x = self.conv2d_1(x)  # -> (batch, 64, 17, 17)
        x = self.conv2d_2(x)  # -> (batch, 128, 15, 15)
        
        # Reshape for Bi-LSTM: each spatial location is a time step => 15*15=225
        x = x.flatten(2).transpose(1, 2)  # -> (batch, 225, 128)
        
        x = self.layer_norm(x)
        
        # Bi-LSTM
        x, _ = self.bilstm_1(x)  # -> (batch, 225, 128)
        x = self.dropout(x)
        x, _ = self.bilstm_2(x)  # -> (batch, 225, 128)
        
        # Take the last time step (225th)
        x = x[:, -1, :]  # -> (batch, 128)
        
        # Classification
        x = self.classifier(x)  # -> (batch, num_classes)
        
        return x
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)