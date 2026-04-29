import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused RMSNorm + QKV projection.
    Performs: RMSNorm(hidden) @ weights
    where RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps)
    """
    def __init__(self, dim, head_dim, eps=1e-6):
        super(Model, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim, head_dim) * 0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden (torch.Tensor): Input tensor of shape (batch, seqlen, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch, seqlen, head_dim).
        """
        h = hidden.float()
        rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + self.eps)
        h_norm = h / rms
        return (h_norm @ self.weight.float()).to(hidden.dtype)


batch = 1
seqlen = 256
dim = 2048
head_dim = 1024

def get_inputs():
    return [torch.randn(batch, seqlen, dim)]

def get_init_inputs():
    return [dim, head_dim]
