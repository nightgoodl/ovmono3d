import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class RotaryPositionEmbedding2D(torch.nn.Module):
    """
    Implements rotary position embeddings based on frequency.
    
    Args:
        frequency (int): Base frequency for the rotary embedding
        offset (float): Offset to apply to the embedding
    """
    def __init__(self, frequency: int = 100, offset: float = 0):
        super().__init__()
        self.frequency = frequency
        self.offset = offset
        
    def forward(self, q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to query tensor.
        
        Args:
            q (torch.Tensor): Query tensor with shape [B, H, N, C]
            pos (torch.Tensor): Position tensor with shape [B, N, 2]
                containing x,y coordinates
                
        Returns:
            torch.Tensor: Query tensor with rotary embedding applied
        """
        # In case q is a different batch size than positions
        assert q.shape[0] == pos.shape[0], f"Expected batch sizes to match but got {q.shape[0]} != {pos.shape[0]}"
        assert q.shape[2] == pos.shape[1], f"Expected sequence lengths to match but got {q.shape[2]} != {pos.shape[1]}"
        
        device, dtype = q.device, q.dtype
        b, h, n, d = q.shape
        
        # For 2D RoPE, we need to handle both x and y dimensions
        x = pos[:, :, 0]  # x coordinates [B, N]
        y = pos[:, :, 1]  # y coordinates [B, N]
        
        # Scale positions by frequency
        if self.frequency > 0:
            x = x * self.frequency + self.offset
            y = y * self.frequency + self.offset
        
        # Ensure dimension is even for pair-wise rotation
        d_rot = (d // 4) * 2  # Use half of d for rotation (quarter for each x,y)
        
        # Generate pair-wise rotation angles for x and y separately
        theta = torch.arange(0, d_rot//2, device=device, dtype=dtype)
        theta = 10000.0 ** (-2.0 * theta / d_rot)
        
        # Apply rotation to pairs for x and y
        x_enc = x.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)  # [B, N, d_rot//2]
        y_enc = y.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)  # [B, N, d_rot//2]
        
        # Get cos and sin values
        cos_x = torch.cos(x_enc)  # [B, N, d_rot//2]
        sin_x = torch.sin(x_enc)  # [B, N, d_rot//2]
        cos_y = torch.cos(y_enc)  # [B, N, d_rot//2]
        sin_y = torch.sin(y_enc)  # [B, N, d_rot//2]
        
        # Add head dimension and expand
        cos_x = cos_x.unsqueeze(1).expand(-1, h, -1, -1)  # [B, H, N, d_rot//2]
        sin_x = sin_x.unsqueeze(1).expand(-1, h, -1, -1)
        cos_y = cos_y.unsqueeze(1).expand(-1, h, -1, -1)
        sin_y = sin_y.unsqueeze(1).expand(-1, h, -1, -1)
        
        # Split query for x and y rotations
        q_x = q[:, :, :, :d_rot//2]
        q_y = q[:, :, :, d_rot//2:d_rot]
        
        # Apply rotations separately for x and y
        q_x_rot = torch.cat([
            q_x * cos_x - q_x * sin_x,
        ], dim=-1)
        
        q_y_rot = torch.cat([
            q_y * cos_y - q_y * sin_y,
        ], dim=-1)
        
        # Combine rotated parts with remaining dimensions
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)
        if d_rot < d:
            q = torch.cat([q_rot, q[:, :, :, d_rot:]], dim=-1)
        else:
            q = q_rot
            
        return q

class PositionGetter(nn.Module):
    """
    Module to compute 2D positions for tokens.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, B, H, W, device=None):
        """
        Generate grid of positions.
        Args:
            B (int): Batch size
            H (int): Height in patches
            W (int): Width in patches
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Position tensor of shape [B, H*W, 2]
        """
        # Create position grid [H, W, 2]
        grid_h = torch.arange(H, device=device)
        grid_w = torch.arange(W, device=device)
        
        # Create meshgrid
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        
        # Flatten and stack to [H*W, 2]
        grid = torch.stack([grid_w.flatten(), grid_h.flatten()], dim=-1).float()
        
        # Normalize positions to [0, 1]
        grid[:, 0] = grid[:, 0] / (W - 1) if W > 1 else 0
        grid[:, 1] = grid[:, 1] / (H - 1) if H > 1 else 0
        
        # Expand for batch dimension [B, H*W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1)
        
        return grid
