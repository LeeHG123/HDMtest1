# functions/dpmpp_sampler.py
from typing import Optional
import torch, tqdm
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from functions.sde import VPSDE1D


def make_dpmpp_scheduler(
    sde: VPSDE1D,
    num_train_steps: int = 1_000,
    device: torch.device | str = "cuda",
    *,
    solver_order: int = 3,          # DPMSolver++ 권장: 3(무조건) / 2(조건)
    solver_type: str = "bh2",       # 확률-흐름 ODE 전용
    lower_order_final: bool = True,
) -> DPMSolverMultistepScheduler:
    """
    HDM VPSDE1D와 동일한 ᾱ(t)·β(t) 테이블을 갖는
    DPMSolver++ Multistep Scheduler 생성.
    """
    T = sde.T
    ts = torch.linspace(
        0.0, T, num_train_steps + 1,
        dtype=torch.float64, device=device,
    )
    log_alpha = sde.marginal_log_mean_coeff(ts)
    alpha_bar = torch.exp(log_alpha)

    betas = 1.0 - torch.clamp(alpha_bar[1:] / alpha_bar[:-1], max=0.9999)
    betas = torch.clamp(betas, 0.0, 0.9999).float().cpu()

    return DPMSolverMultistepScheduler(
        trained_betas       = betas,
        num_train_timesteps = len(betas),
        algorithm_type      = "dpmsolver++",   #★  DPM-Solver++
        solver_order        = solver_order,
        solver_type         = solver_type,
        lower_order_final   = lower_order_final,
        prediction_type     = "epsilon",
        thresholding        = False,
        sample_max_value    = 1.0,
    )


@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    x_t0: torch.Tensor,      
    x_coord: torch.Tensor,      
    *,
    batch_size: Optional[int] = None,
    data_dim: Optional[int] = None,
    device: torch.device | str = "cuda",
    inference_steps: int = 500,
    fp16: bool = False,
    progress: bool = True,
):
    """DPMSolver++ ODE (확률-흐름) 샘플링."""
    scheduler = make_dpmpp_scheduler(sde, device=device)
    scheduler.set_timesteps(inference_steps, device=device)

    if x_t0 is None:
        if batch_size is None or data_dim is None:
            raise ValueError("batch_size·data_dim 필요")
        
        is_model_2d = hasattr(model, 'n_dim') and model.n_dim == 2
        if is_model_2d:
            x = torch.randn(batch_size, data_dim, data_dim, device=device)
        else:
            x = torch.randn(batch_size, data_dim, device=device)
    else:
        x = x_t0.to(device)

    x_coord = x_coord.to(device)
    batch_size = x.shape[0]
    is_2d = (x.dim() == 3)

    model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.no_grad

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps, disable=not progress)):
        with autocast():
            t_cont = t.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
            t_vec  = t_cont.repeat(batch_size).to(device).to(x.dtype)
            if is_2d:
                x_coord_ch = x_coord.permute(0, 3, 1, 2)
                model_input = torch.cat([x.unsqueeze(1), x_coord_ch], dim=1)
                score = model(model_input, t_vec).squeeze(1)
            else:
                model_input = torch.cat([x.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
                score = model(model_input, t_vec)                       # ∇ₓ log p(xᵗ)
        epsilon = -score                                    # prediction_type="epsilon"
        x = scheduler.step(epsilon, t, x).prev_sample

        # 동일한 L²-클리핑 (Quadratic 데이터셋 기준)
        flat = x.view(x.size(0), -1)
        norm = flat.norm(dim=1, keepdim=True)
        if is_2d:
            H, W = x.shape[1], x.shape[2]
            threshold = 10.0 * math.sqrt(H * W)
        else:
            threshold = 10.0        
        mask = (norm > threshold).squeeze(1)          
        flat[mask] = flat[mask] / norm[mask] * threshold
        x = flat.view_as(x)

    return x
