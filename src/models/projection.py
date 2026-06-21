import torch
import torch.nn as nn

class ProjectionBridge(nn.Module):
    """
    Projection bridge to map continuous audio encoder hidden states
    to the compact decoder's input embedding space.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, downsample_factor: int = 1):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Linear projection option
        if downsample_factor == 1:
            self.projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            # Convolutional downsampling 
            # Kernel size and stride equal to downsample factor
            self.projection = nn.Conv1d(
                in_channels=encoder_dim,
                out_channels=decoder_dim,
                kernel_size=downsample_factor,
                stride=downsample_factor
            )
            
        self.layer_norm = nn.LayerNorm(decoder_dim)
        
    def forward(self, encoder_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_states: Tensor of shape (B, T_audio, D_enc)
            
        Returns:
            Projected states: Tensor of shape (B, T_proj, D_dec)
        """
        if self.downsample_factor == 1:
            x = self.projection(encoder_states)
        else:
            # Conv1d expects (B, Channels, Length)
            x = encoder_states.transpose(1, 2)
            x = self.projection(x)
            # Transpose back to (B, Length, Channels)
            x = x.transpose(1, 2)
            
        return self.layer_norm(x)
