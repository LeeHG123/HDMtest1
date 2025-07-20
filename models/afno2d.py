import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import get_timestep_embedding, Lifting, Projection, default_init, skip_connection

class AFNO2DBlock(nn.Module):
    """
    2D 데이터용 Adaptive Fourier Neural Operator 블록.
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, 
                 hard_thresholding_fraction=1.0, hidden_size_factor=2, scale=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = scale

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        # 2D FFT
        x = torch.fft.rfft2(x, norm="ortho")
        
        # 저주파수 모드 선택
        kept_h = int(x.shape[-2] * self.hard_thresholding_fraction)
        kept_w = int(x.shape[-1] * self.hard_thresholding_fraction)
        
        x_low = x[:, :, :kept_h, :kept_w]

        x_reshaped = x_low.reshape(B, self.num_blocks, self.block_size, kept_h, kept_w)
        
        x_permuted = x_reshaped.permute(0, 3, 4, 1, 2)

        # BlockMLP in Frequency Domain
        o1_real = F.relu(
            torch.einsum('...si,sio->...so', x_permuted.real, self.w1[0]) -
            torch.einsum('...si,sio->...so', x_permuted.imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum('...si,sio->...so', x_permuted.imag, self.w1[0]) +
            torch.einsum('...si,sio->...so', x_permuted.real, self.w1[1]) +
            self.b1[1]
        )
        o2_real = (
            torch.einsum('...so,soi->...si', o1_real, self.w2[0]) -
            torch.einsum('...so,soi->...si', o1_imag, self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('...so,soi->...si', o1_imag, self.w2[0]) +
            torch.einsum('...so,soi->...si', o1_real, self.w2[1]) +
            self.b2[1]
        )

        x_processed_permuted = torch.stack([o2_real, o2_imag], dim=-1)
        x_processed_permuted = F.softshrink(x_processed_permuted, lambd=self.sparsity_threshold)
        x_processed_permuted = torch.view_as_complex(x_processed_permuted)
        
        x_processed = x_processed_permuted.permute(0, 3, 4, 1, 2)
        
        # [수정] 블록 차원(num_blocks, block_size)을 채널 차원(C)으로 다시 합칩니다.
        # (B, num_blocks, block_size, H, W) -> (B, C, H, W)
        x_processed = x_processed.reshape(B, C, kept_h, kept_w)

        # 원래 주파수 텐서 크기로 복원
        x_out = torch.zeros_like(x)
        x_out[:, :, :kept_h, :kept_w] = x_processed # 이제 형태가 일치합니다.

        # Inverse 2D FFT
        x = torch.fft.irfft2(x_out, s=(H, W), norm="ortho")
        x = x.type(dtype)

        return x + bias

class AFNO2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.n_dim = 2 # 2D 모델
        self.act = nn.SiLU()

        self.hidden_channels = self.config.hidden_channels
        self.lifting_channels = self.config.lifting_channels
        self.projection_channels = self.config.projection_channels
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.n_layers = self.config.n_layers

        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)
        
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=self.out_channels, hidden_channels=self.projection_channels, n_dim=self.n_dim)

        self.afno_blocks = nn.ModuleList([
            AFNO2DBlock(
                hidden_size=self.hidden_channels,
                num_blocks=self.config.num_blocks,
                sparsity_threshold=self.config.sparsity_threshold,
                hidden_size_factor=self.config.hidden_size_factor,
                hard_thresholding_fraction=self.config.hard_thresholding_fraction 
            ) for _ in range(self.n_layers)
        ])
        
        self.norms = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels) for _ in range(self.n_layers)])
        self.skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, n_dim=self.n_dim, type='soft-gating') for _ in range(self.n_layers)])

    def forward(self, x, t):
        # x shape: (B, C, H, W)
        x = self.lifting(x)

        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](self.act(temb))
        
        # Add time embedding to spatial dimensions
        x = x + temb.unsqueeze(-1).unsqueeze(-1)

        for i in range(self.n_layers):
            x_skip = self.skips[i](x)
            
            x = self.norms[i](x)
            x = self.afno_blocks[i](x)
            
            x = x + x_skip

        x = self.projection(x)
        return x