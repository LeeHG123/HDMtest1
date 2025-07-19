import torch
import torch.nn.functional as F

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)
    noise = sde.marginal_std(e, t)

    x_t = x_mean + noise
    score = -noise

    output = model(x_t, t)

    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)

    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e

    output = model(x_t, t.float())

    loss = (output - score).square().sum(dim=(1)).mean(dim=0)
    return loss

def hilbert_sobolev_loss_fn(model, sde, x_0, t, e, x_coord, sobolev_weight=1.0):
    """
    Computes the H¹ Sobolev norm-based loss for smoother 1D function generation.
    The loss is a weighted sum of the L² norm and the L² norm of the first derivative.
    The derivative is computed efficiently in the Fourier domain.

    Loss = ||output - score||_L²² + λ * ||∇(output - score)||_L²²
    
    Args:
        model: The score network.
        sde: The SDE object.
        x_0: The original data (ground truth functions).
        t: Timesteps.
        e: Noise sampled from Hilbert space.
        sobolev_weight (float): The weight (λ) for the derivative term.
    
    Returns:
        A scalar tensor representing the Sobolev loss.
    """
    # 1. Get model output (predicted score)
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)
    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e
    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1) # shape: (B, 2, N)
    output = model(model_input, t.float())

    # 2. Calculate the difference between prediction and target
    diff = output - score  # Shape: (batch_size, num_points)

    # 3. Calculate the L² norm component of the loss
    l2_loss_term = diff.square().sum(dim=1)

    # 4. Calculate the H¹ norm component (derivative term) using Fourier differentiation
    num_points = diff.shape[1]
    
    # Apply Real Fast Fourier Transform for real-valued signals
    diff_fft = torch.fft.rfft(diff, dim=1)
    
    # Get the corresponding frequencies.
    freqs = torch.fft.rfftfreq(num_points, device=diff.device)
    freqs_sq = freqs.square()
    
    # Weight the squared magnitudes of FFT coefficients by squared frequencies
    h1_loss_term = (diff_fft.abs().square() * freqs_sq[None, :]).sum(dim=1)

    # 5. Combine the terms to get the final Sobolev loss
    sobolev_loss = (l2_loss_term + sobolev_weight * h1_loss_term).mean(dim=0)

    return sobolev_loss

def hilbert_sobolev_loss_fn_2d(model, sde, x_0, t, e, x_coord, sobolev_weight=1.0):
    """
    Computes the H¹ Sobolev norm-based loss for 2D function generation.
    Loss = ||output - score||_L²² + λ * ||∇(output - score)||_L²²
    
    Args:
        model: The score network for 2D data.
        sde: The SDE object.
        x_0: The original 2D data (B, H, W).
        t: Timesteps (B,).
        e: Noise from Hilbert space (B, H, W).
        x_coord: Coordinates of the grid (B, H, W, 2).
        sobolev_weight (float): Weight for the derivative term.
    
    Returns:
        A scalar tensor for the 2D Sobolev loss.
    """
    # 1. 모델 출력(예측 스코어) 얻기
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)
    
    # 텐서 형태를 (B, H, W)에 맞게 조정
    x_t = x_0 * x_mean[:, None, None] + e * noise[:, None, None]
    score = -e

    # 모델 입력 준비: (B, 1, H, W) 데이터와 (B, 2, H, W) 좌표 결합 -> (B, 3, H, W)
    x_t_ch = x_t.unsqueeze(1)
    x_coord_ch = x_coord.permute(0, 3, 1, 2) # (B, H, W, 2) -> (B, 2, H, W)
    model_input = torch.cat([x_t_ch, x_coord_ch], dim=1)
    
    output = model(model_input, t.float()).squeeze(1) # (B, 1, H, W) -> (B, H, W)

    # 2. 예측과 타겟의 차이 계산
    diff = output - score  # Shape: (B, H, W)

    # 3. L² norm 손실 계산
    l2_loss_term = diff.square().sum(dim=(1, 2))

    # 4. H¹ norm (미분 항) 손실 계산 (2D 푸리에 미분)
    H, W = diff.shape[1], diff.shape[2]
    
    diff_fft = torch.fft.rfftn(diff, dim=(1, 2))
    
    freqs_y = torch.fft.fftfreq(H, device=diff.device)
    freqs_x = torch.fft.rfftfreq(W, device=diff.device)
    
    freqs_y_grid, freqs_x_grid = torch.meshgrid(freqs_y, freqs_x, indexing='ij')
    freqs_sq = freqs_y_grid.square() + freqs_x_grid.square()
    
    h1_loss_term = (diff_fft.abs().square() * freqs_sq[None, :, :]).sum(dim=(1, 2))

    # 5. 최종 Sobolev 손실 결합
    sobolev_loss = (l2_loss_term + sobolev_weight * h1_loss_term).mean(dim=0)

    return sobolev_loss