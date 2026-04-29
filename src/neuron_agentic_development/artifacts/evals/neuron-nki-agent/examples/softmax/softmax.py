import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Softmax along the last dimension of a 2D tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (rows, cols).
        Returns:
            torch.Tensor: Output tensor of shape (rows, cols) with softmax applied along dim=-1.
        """
        return torch.softmax(x, dim=-1)


rows = 1024
cols = 8192

def get_inputs():
    return [torch.randn(rows, cols)]

def get_init_inputs():
    return []
