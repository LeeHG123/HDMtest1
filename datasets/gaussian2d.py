import torch
import math

class Gaussian2DDataset(torch.utils.data.Dataset):
    """
    2차원 표준 정규분포의 로그 확률 밀도 함수(log-PDF)를 생성합니다.
    y = log φ(x, y) + ε  (φ: 2D 표준정규 PDF)
    ─────────────────────────────────────
    - 저장 형태  : z-score 정규화된 2D log-pdf
    - 역변환 함수: self.inverse_transform(...)
    """

    def __init__(self, num_data: int, num_points: int, seed: int = 42):
        """
        Args:
            num_data (int): 생성할 데이터 샘플의 수
            num_points (int): 각 축의 그리드 포인트 수 (결과적으로 num_points x num_points 그리드 생성)
            seed (int): 재현성을 위한 랜덤 시드
        """
        super().__init__()
        torch.manual_seed(seed)

        # 1) 2D 균일 그리드 (x, y) ∈ [-10,10] x [-10,10]
        grid_coords = torch.linspace(-10., 10., steps=num_points)
        x, y = torch.meshgrid(grid_coords, grid_coords, indexing='ij')  # (N, N), (N, N)
        
        # 좌표를 (B, H, W, 2) 형태로 저장
        self.coords = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(num_data, 1, 1, 1) # (B, N, N, 2)

        # 2) 2D log-pdf 계산: log(PDF) = -0.5*(x² + y²) - log(2π)
        log_phi = -0.5 * (x**2 + y**2) - math.log(2 * math.pi)  # (N, N)
        log_phi = log_phi.unsqueeze(0).repeat(num_data, 1, 1)      # (B, N, N)

        # 3) ε ~ 𝒩(0, 10⁻³) 노이즈 추가
        eps = torch.randn_like(log_phi) * 1e-2
        log_phi_noisy = log_phi + eps

        # 4) 데이터셋 전체에 대해 Z-정규화
        self.mean = log_phi_noisy.mean()
        self.std = log_phi_noisy.std()
        self.dataset = (log_phi_noisy - self.mean) / self.std  # (B, N, N)

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, idx):
        # 좌표는 (H, W, 2), 데이터는 (H, W, 1) 형태로 반환
        return (
            self.coords[idx],
            self.dataset[idx].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        입력: z-score 정규화된 2D log-pdf (..., H, W)
        반환: 원래 스케일의 2D PDF 값 (..., H, W)
        """
        y_log = y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)
        return torch.exp(y_log)