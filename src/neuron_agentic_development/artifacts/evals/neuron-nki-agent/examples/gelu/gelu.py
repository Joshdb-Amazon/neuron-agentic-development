import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    GELU activation applied element-wise to a 2D tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (rows, cols).
        Returns:
            torch.Tensor: Output tensor of shape (rows, cols) with GELU applied.
        """
        return F.gelu(x)


rows = 2048
cols = 8192

def get_inputs():
    return [torch.randn(rows, cols)]

def get_init_inputs():
    return []
