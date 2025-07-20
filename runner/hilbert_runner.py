import os
import logging
from scipy.spatial import distance
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위한 라이브러리
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import numpy as np                      # 좌표 그리드 생성을 위해 추가
import math

from evaluate.power import calculate_ci
from datasets import data_scaler, data_inverse_scaler

from collections import OrderedDict

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from models.afno import AFNO
from models.afno2d import AFNO2D 
from models import *

from functions.utils import *
from functions.loss import hilbert_loss_fn, hilbert_sobolev_loss_fn, hilbert_sobolev_loss_fn_2d
from functions.sde import VPSDE1D
from functions.sampler import sampler
from functions.unipc_sampler import sample_probability_flow_ode as unipc_sample_ode
from functions.dpmpp_sampler import sample_probability_flow_ode as dpmpp_sample_ode
from functions.deis_sampler  import sample_probability_flow_ode as deis_sample_ode
from functions.tsit5_sampler import sample_probability_flow_ode as tsit5_sample_ode

torch.autograd.set_detect_anomaly(True)

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function for both 1D and 2D """
    x1_np = x1.cpu().numpy()
    x2_np = x2.cpu().numpy()

    # 입력 차원에 따라 거리 계산
    if x1_np.shape[-1] == 1: # 1D case
        D = distance.cdist(x1_np / hyp['len'], x2_np / hyp['len'], 'sqeuclidean')
    else: # 2D case
        D = distance.cdist(x1_np / hyp['len'], x2_np / hyp['len'], 'euclidean')
        D = D**2 # squared euclidean

    K = hyp['gain'] * np.exp(-0.5 * D) # SE 커널 공식 수정
    return torch.from_numpy(K).to(torch.float32).to(x1.device)

class HilbertNoise:
    def __init__(self, grid, x=None, hyp_len=1.0, hyp_gain=1.0, use_truncation=False):
        x = torch.linspace(-10, 10, grid)
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        x = torch.unsqueeze(x, dim=-1)
        self.x = x
        if x is not None:
            self.x=x

        K = kernel_se(x, x, self.hyp)
        K = K.cpu().numpy()
        eig_val, eig_vec = np.linalg.eigh(K + 1e-6 * np.eye(K.shape[0], K.shape[0]))

        self.eig_val = torch.from_numpy(eig_val)
        self.eig_vec = torch.from_numpy(eig_vec).to(torch.float32)
        self.D = torch.diag(self.eig_val).to(torch.float32)
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D))

    def sample(self, size):
        size = list(size)  # batch*grid
        x_0 = torch.randn(size)

        output = (x_0 @ self.M.transpose(0, 1))  # batch grid x grid x grid
        return output  # bath*grid

    def free_sample(self, free_input):  # input (batch,grid)

        y = torch.randn(len(free_input), self.x.shape[0]) @ self.eig_vec.T @ kernel_se(self.x, free_input[0].unsqueeze(-1), self.hyp)
        return y

class HilbertNoise2D:
    """Hilbert space noise generation for 2D data."""
    def __init__(self, grid_size, hyp_len=1.0, hyp_gain=1.0, device='cuda'):
        """
        Args:
            grid_size (int): 각 축의 그리드 포인트 수 (결과: grid_size x grid_size).
            hyp_len (float): Squared-exponential 커널의 길이 스케일.
            hyp_gain (float): Squared-exponential 커널의 게인.
            device (str): 연산을 수행할 장치.
        """
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        self.device = device

        # 1. 2D 좌표 그리드 생성 및 평탄화
        # 결과적으로 self.x는 (grid_size*grid_size, 2) 형태의 텐서가 됨
        coords = torch.linspace(-10., 10., steps=grid_size, device=device)
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        self.x = torch.stack([x.flatten(), y.flatten()], dim=1) # Shape: (N*N, 2)

        # 2. 2D SE 커널 계산
        # kernel_se 함수는 입력 좌표의 차원에 따라 자동으로 1D/2D를 처리
        K = kernel_se(self.x, self.x, self.hyp)
        
        # 3. Eigendecomposition
        eig_val, eig_vec = torch.linalg.eigh(K + torch.eye(K.shape[0], device=device))
        
        self.eig_val = eig_val.to(torch.float32)
        self.eig_vec = eig_vec.to(torch.float32)
        self.D = torch.diag(self.eig_val)
        
        # 4. 변환 행렬 M 계산
        self.M = torch.matmul(self.eig_vec, torch.sqrt(torch.diag(torch.clamp(self.eig_val, min=0.0))))

    def sample(self, size):
        """
        고정된 그리드에서 Hilbert 공간 노이즈를 샘플링합니다.
        
        Args:
            size (tuple): (batch_size, height, width) 형태의 크기.
        
        Returns:
            torch.Tensor: (batch, H, W) 모양의 공간적 상관관계가 있는 노이즈.
        """
        batch_size, H, W = size
        num_points = H * W
        
        # i.i.d. 가우시안 노이즈 생성
        z = torch.randn(batch_size, num_points, device=self.device)

        # 변환 행렬 M을 사용하여 상관관계가 있는 노이즈로 변환
        output = z @ self.M.T  # (B, N*N) @ (N*N, N*N) -> (B, N*N)
        
        # 원래 2D 이미지 형태로 복원
        return output.view(batch_size, H, W)

    def free_sample(self, free_input_coords):
        """
        임의의 해상도(resolution-free)에서 노이즈를 샘플링합니다.
        
        Args:
            free_input_coords (torch.Tensor): (B, H_new, W_new, 2) 모양의 새로운 좌표.
        
        Returns:
            torch.Tensor: (B, H_new, W_new) 모양의 새로운 좌표에 해당하는 노이즈.
        """
        batch_size, H_new, W_new, _ = free_input_coords.shape
        num_new_points = H_new * W_new
        
        batch_samples = []
        for i in range(batch_size):
            # i번째 샘플의 새로운 좌표를 평탄화: (H_new * W_new, 2)
            coords_new = free_input_coords[i].view(-1, 2).to(self.device)

            # 1. 원본 그리드와 새로운 좌표 간의 Cross-covariance 커널 계산
            # K_cross: (N*N, H_new*W_new)
            K_cross = kernel_se(self.x, coords_new, self.hyp)
            
            # 2. 새로운 좌표에 대한 투영 기저(basis) 계산
            # N = L^T * K_cross, 여기서 L은 eig_vec
            # N: (N*N, H_new*W_new)
            N = self.eig_vec.T @ K_cross

            # 3. 표준 정규분포에서 랜덤 계수 z를 샘플링
            z = torch.randn(1, self.eig_val.shape[0], device=self.device) # (1, N*N)

            # 4. z를 새로운 기저 N에 투영하여 샘플 생성
            y_flat = z @ N  # (1, N*N) @ (N*N, H_new*W_new) -> (1, H_new*W_new)

            # 5. 결과를 2D 형태로 복원하여 리스트에 추가
            y = y_flat.view(H_new, W_new)
            batch_samples.append(y)
            
        # 배치 차원을 따라 모든 샘플을 하나로 합침
        return torch.stack(batch_samples, dim=0)    

class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- 2D/1D 분기 처리 ---
        self.is_2d = (config.data.modality == "2D")          # <‑‑ 새 플래그
        if self.is_2d:
            self.W = HilbertNoise2D(
                grid_size=config.data.dimension,
                hyp_len=config.data.hyp_len,
                hyp_gain=config.data.hyp_gain,
                device=self.device,
            )
            logging.info("Initialized **HilbertNoise2D**.")
        else:
            self.W = HilbertNoise(
                grid=config.data.dimension,
                hyp_len=config.data.hyp_len,
                hyp_gain=config.data.hyp_gain,
            )
            logging.info("Initialized **HilbertNoise (1D)**.")

        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.sde = VPSDE1D(schedule='cosine')
        self.dataset = dataset
        self.test_dataset = test_dataset
        
    def validate(self, model, val_loader, tb_logger, step):
        """Validation function to compute validation loss"""
        model.eval()
        val_losses = [] # 이 리스트는 그대로 유지합니다.
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= 10:
                    break
                    
                if self.is_2d:
                    x_coord = x.to(self.device)
                    y_norm  = y.to(self.device).squeeze(-1)
                    x_coord_norm = x_coord / 10.0
                else:
                    x_coord = x.to(self.device).squeeze(-1)
                    y_norm  = y.to(self.device).squeeze(-1)
                    x_coord_norm = x_coord / 10.0
                
                if self.config.data.dataset == 'Melbourne':
                    y = data_scaler(y)
                
                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                if self.is_2d:
                    e = self.W.sample((y_norm.size(0), y_norm.size(1), y_norm.size(2))).to(self.device)
                else:
                    e = self.W.sample(y_norm.shape).to(self.device).squeeze(-1)
              
                if self.is_2d:
                    batch_loss = hilbert_sobolev_loss_fn_2d(
                        model, self.sde, y_norm, t, e, x_coord_norm,
                        sobolev_weight=self.config.training.sobolev_weight
                    )
                else:
                    batch_loss = hilbert_sobolev_loss_fn(
                        model, self.sde, y_norm, t, e, x_coord_norm,
                        sobolev_weight=self.config.training.sobolev_weight
                    )
                # 'batch_loss'의 스칼라 값을 'val_losses' 리스트에 추가합니다.
                val_losses.append(batch_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        tb_logger.add_scalar("val_loss", avg_val_loss, global_step=step)
        
        model.train()
        return avg_val_loss

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if args.distributed:
            sampler = DistributedSampler(self.dataset, shuffle=True,
                                     seed=args.seed if args.seed is not None else 0)
        else:
            sampler = None
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        )
        
        # Validation loader
        val_loader = data.DataLoader(
            self.test_dataset,
            batch_size=config.training.val_batch_size,
            num_workers=config.data.num_workers,
            shuffle=False
        )

        # Model
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)  
        elif config.model.model_type == "AFNO2D":          # NEW
            model = AFNO2D(config)
        elif config.model.model_type == "AFNO":
            model = AFNO(config)        
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],)
                                                            #   find_unused_parameters=True)
        logging.info("Model loaded.")

        # Optimizer, LR scheduler
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=2000000,
        # )

        start_epoch, step = 0, 0
        # if args.resume:
        #     states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=self.device)
        #     model.load_state_dict(states[0], strict=False)
        #     start_epoch = states[2]
        #     step = states[3]

        for epoch in range(config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                if self.is_2d:
                    x_coord = x.to(self.device)                 # (B,H,W,2)
                    y_norm  = y.to(self.device).squeeze(-1)     # (B,H,W)
                    x_coord_norm = x_coord / 10.0
                else:
                    x_coord = x.to(self.device).squeeze(-1)     # (B,N)
                    y_norm  = y.to(self.device).squeeze(-1)     # (B,N)
                    x_coord_norm = x_coord / 10.0

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                if self.is_2d:
                    e = self.W.sample((y_norm.size(0), y_norm.size(1), y_norm.size(2))).to(self.device)
                else:
                    e = self.W.sample(y_norm.shape).to(self.device)

                if self.is_2d:
                    loss_score = hilbert_sobolev_loss_fn_2d(
                        model, self.sde, y_norm, t, e, x_coord_norm,
                        sobolev_weight=self.config.training.sobolev_weight,
                    )
                else:
                    loss_score = hilbert_sobolev_loss_fn(
                        model, self.sde, y_norm, t, e, x_coord_norm,
                        sobolev_weight=self.config.training.sobolev_weight,
                    )
                # 정규화 보조-손실
                if self.config.data.dataset == "Gaussian":
                    dx = (self.dataset.x[0, 1] - self.dataset.x[0, 0]).abs().to(self.device)

                    t0 = torch.zeros(y.shape[0], device=self.device)
                    model_input_t0 = torch.cat([y.unsqueeze(1), x_coord_norm.unsqueeze(1)], dim=1)
                    score = model(model_input_t0, t0)

                    log_pdf_pred = torch.cumsum(score * dx, dim=1)
                    
                    log_Z_pred = torch.logsumexp(log_pdf_pred, dim=1) + torch.log(dx)
                    Z_pred = torch.exp(log_Z_pred)
                    
                    norm_loss = (Z_pred - 1.0).pow(2).mean()

                    λ_max = config.training.lambda_norm
                    λ     = λ_max * min(1.0, step / 1000)

                    loss = loss_score + λ * norm_loss
                else:
                    loss = loss_score
                tb_logger.add_scalar("train_loss", torch.abs(loss), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()
                # lr_scheduler.step()

                # Validation
                if step % config.training.val_freq == 0:
                    val_loss = self.validate(model, val_loader, tb_logger, step)
                    if args.local_rank == 0:
                        logging.info(f"step: {step}, val_loss: {val_loss}")

                if step % config.training.ckpt_store == 0:
                    self.ckpt_dir = os.path.join(args.log_path, f'ckpt_step_{step}.pth')
                    torch.save(model.state_dict(), self.ckpt_dir)
                    
                    # Also save the latest checkpoint as ckpt.pth
                    latest_ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), latest_ckpt_dir)

                data_start = time.time()

    def visualize(self, x_true, x_pred, save_path):
            if self.is_2d:
                # 3D 표면 플롯으로 변경
                fig = plt.figure(figsize=(14, 6))

                # --- 1. 좌표 그리드 생성 ---
                # 데이터의 shape으로부터 그리드 크기(H, W)를 가져옵니다.
                # x_true는 (B, H, W) 형태이므로, x_true.shape[1:]을 사용합니다.
                H, W = x_true.shape[1:]
                x_coords = np.linspace(-10, 10, W)
                y_coords = np.linspace(-10, 10, H)
                X, Y = np.meshgrid(x_coords, y_coords)

                # --- 2. Ground Truth(GT) 플롯 ---
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.set_title('GT')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('PDF Value')
                
                # 여러 GT 샘플을 반투명하게 겹쳐 그리기 (최대 10개)
                num_samples_to_plot = min(10, x_true.shape[0])
                for i in range(num_samples_to_plot):
                    Z1 = x_true[i].cpu().numpy()
                    ax1.plot_surface(X, Y, Z1, color='c', edgecolor='none', alpha=0.15)

                # --- 3. Generated 샘플 플롯 ---
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.set_title('Generated')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('PDF Value')

                # 여러 생성 샘플을 반투명하게 겹쳐 그리기 (최대 10개)
                num_samples_to_plot = min(10, x_pred.shape[0])
                for i in range(num_samples_to_plot):
                    Z2 = x_pred[i].cpu().numpy()
                    # 마지막 샘플에만 cmap을 적용하여 대표 컬러바를 생성
                    if i == num_samples_to_plot - 1:
                        surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none', alpha=0.3)
                    else:
                        ax2.plot_surface(X, Y, Z2, color='m', edgecolor='none', alpha=0.15)
                fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10)

            else:
                # 1D 데이터에 대한 기존 플롯 코드는 그대로 둡니다.
                # 여러 샘플을 겹쳐 그리도록 수정
                num_samples_to_plot = min(10, x_true.shape[0])
                for i in range(num_samples_to_plot):
                    # 첫 번째 샘플에만 레이블을 추가하여 범례가 중복되지 않도록 함
                    label_gt = 'GT' if i == 0 else None
                    label_gen = 'Generated' if i == 0 else None
                    plt.plot(x_true[i].cpu(), 'r-', label=label_gt, alpha=0.5)
                    plt.plot(x_pred[i].cpu(), 'b--', label=label_gen, alpha=0.5)
                plt.legend()
                
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()        

    def sample(self, score_model=None):
        args, config = self.args, self.config

        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional,)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "AFNO2D":          # NEW
            model = AFNO2D(config)
        elif config.model.model_type == "AFNO":
            model = AFNO(config)       
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif "ckpt_dir" in config.model.__dict__.keys():
            # Check if specific checkpoint step is requested
            if args.ckpt_step is not None:
                ckpt_path = os.path.join(args.log_path, f'ckpt_step_{args.ckpt_step}.pth')
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                    logging.info(f"Using checkpoint from step {args.ckpt_step}: {ckpt_path}")
                else:
                    logging.warning(f"Checkpoint for step {args.ckpt_step} not found: {ckpt_path}")
                    logging.info("Falling back to latest checkpoint")
                    ckpt_path = os.path.join(args.log_path, 'ckpt.pth')
                    if os.path.exists(ckpt_path):
                        ckpt_dir = ckpt_path
                    else:
                        ckpt_dir = config.model.ckpt_dir
            else:
                # First try the latest checkpoint from training
                ckpt_path = os.path.join(args.log_path, 'ckpt.pth')
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                else:
                    ckpt_dir = config.model.ckpt_dir
            
            states = torch.load(
                ckpt_dir,
                map_location=config.device,
            )

            if args.distributed:
                state_dict = OrderedDict()
                for k, v in states.items():
                    if 'module' in k:
                        name = k[7:]
                        state_dict[name] = v
                    else:
                        state_dict[k] = v

                model.load_state_dict(state_dict)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            else:
                model.load_state_dict(states, strict=False)
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        logging.info("Done loading model")
        model.eval()

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)
                        
        if self.is_2d:
            # 2D Resolution-free 좌표 및 GT 생성
            new_res = 50
            batch_size = config.sampling.batch_size
            grid_coords = torch.linspace(-10., 10., steps=new_res)
            x_new, y_new = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
            free_input_coords = torch.stack([x_new, y_new], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            log_phi_new = -0.5 * (x_new**2 + y_new**2) - math.log(2 * math.pi)
            y00_norm = (log_phi_new - self.dataset.mean) / self.dataset.std
            y00 = y00_norm.unsqueeze(0).repeat(batch_size, 1, 1)
        else: # 1D Case
            # 1D Resolution-free 좌표 및 GT 생성
            free_input = torch.rand((config.sampling.batch_size, y_0_fixed.shape[1])) * 20 - 10
            free_input = torch.sort(free_input)[0]
            if config.data.dataset == 'Quadratic':
                a = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, 100) * 2 - 1
                eps = torch.normal(mean=0., std=1., size=(free_input.shape[0], 1)).repeat(1, 100)
                y00 = a * (free_input ** 2) + eps
            elif config.data.dataset == 'Gaussian':
                phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * free_input ** 2)
                eps = torch.normal(mean=0., std=0.01, size=(free_input.shape[0], 1)).repeat(1, 100) * 0.1
                y00 = phi + eps

        y = None # 생성된 샘플을 저장할 변수

        # <<< --- 통합된 샘플링 및 결과 처리 로직 --- >>>
        if self.args.sample_type.endswith("_ode"):
            with torch.no_grad():
                # 1. 샘플링에 필요한 초기값 준비
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                
                if self.is_2d and free_input_coords is not None:
                    # Case 1: 2D Resolution-free
                    x_coord_for_sampling = free_input_coords
                    x_t = self.W.free_sample(x_coord_for_sampling)
                else: # Case 2: 1D Resolution-free 또는 고정 그리드
                    if free_input is not None: # 1D Resolution-free
                        x_coord_for_sampling = free_input.unsqueeze(-1)
                        x_t = self.W.free_sample(free_input)
                    else: # 고정 그리드 (1D or 2D)
                        x_coord_for_sampling = x_0_fixed
                        x_t = self.W.sample(y_0_fixed.shape)

                # 초기 노이즈 스케일링
                x_t = x_t.to(self.device)
                norm_dims = list(range(1, x_t.dim()))
                x_t = x_t / torch.std(x_t, dim=norm_dims, keepdim=True)
                x_t = x_t * self.sde.marginal_std(t).view([-1] + [1] * len(norm_dims))

                # 좌표 정규화
                x_coord_norm = x_coord_for_sampling.to(self.device) / 10.0
                if not self.is_2d:
                    x_coord_norm = x_coord_norm.squeeze(-1)

                # 2. 선택된 ODE 샘플러 호출
                sampler_fn = {
                    "unipc_ode": unipc_sample_ode,
                    "dpmpp_ode": dpmpp_sample_ode,
                    "deis_ode": deis_sample_ode,
                    "tsit5_ode": tsit5_sample_ode,
                }[self.args.sample_type]

                y = sampler_fn(
                    model, self.sde,
                    x_t0=x_t,
                    x_coord=x_coord_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                )
            
            # 3. 결과 처리 및 시각화 (y가 정의된 동일 블록 내에서 수행)
            if y is not None:
                if self.is_2d:
                    # 2D 데이터 결과 처리
                    y_plot = self.test_dataset.inverse_transform(y).cpu()
                    y_true_vis = self.test_dataset.inverse_transform(y00).cpu()

                    # 시각화
                    save_path = f"visualization_{self.args.sample_type}_resfree.png"
                    self.visualize(
                        x_true=y_true_vis,
                        x_pred=y_plot,
                        save_path=save_path,
                    )
                    print(f"Saved resolution-free plot to {save_path}")
                    
                    # Power 계산
                    y_pow_flat = y_plot.flatten(1)
                    y_gt_flat = y_true_vis.flatten(1)
                    n_tests = config.sampling.batch_size // 10
                    power_res = calculate_ci(y_pow_flat, y_gt_flat, n_tests=n_tests)
                    print(f"[{self.args.sample_type}] resolution-free power(avg 30) = {power_res}")
                else:
                    # 1D 데이터 결과 처리
                    scale = 50.0 if config.data.dataset == "Quadratic" else 1.0
                    y_plot = (y * scale).cpu()
                    y_gt = y00.cpu()
                    
                    # Power 계산
                    n_tests = y_plot.shape[0] // 10
                    power_res = calculate_ci(y_plot, y_gt, n_tests=n_tests)
                    print(f"[{self.args.sample_type}] resolution-free power(avg 30) = {power_res}")
                    
                    # 시각화
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    y_0_plot = (y_0_fixed * scale).cpu()
                    for i in range(min(10, y_0_plot.shape[0])):
                        ax[0].plot(x_0_fixed[i].squeeze(-1).cpu(), y_0_plot[i], color="k", alpha=.7)
                    ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

                    for i in range(y_plot.shape[0]):
                        ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
                    ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

                    fig.suptitle(f"{self.args.sample_type.upper()} (NFE={self.args.nfe})", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"visualization_{self.args.sample_type}.png")
                    print(f"Saved plot fig to visualization_{self.args.sample_type}.png")
                    plt.close()
                              
         # ──── ①-A UniPC 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "unipc_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot 
            y_gt  = y00.cpu()                   # (B,100)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[UniPC] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"UniPC-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_unipc.png")
            print("Saved plot fig to visualization_unipc.png")
            plt.clf(); plt.figure()

        # ──── ①-A Dpmpp 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "dpmpp_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[Dpmpp] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Dpmpp-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_dpmpp.png")
            print("Saved plot fig to visualization_dpmpp.png")
            plt.clf(); plt.figure()           

        # ──── ①-A DEIS 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "deis_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[DEIS] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"DEIS-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_deis.png")
            print("Saved plot fig to visualization_deis.png")
            plt.clf(); plt.figure()

        # ──── ①-A Tsit5 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Quadratic"]:
            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[Tsit5] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Tsit5-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_tsit5.png")
            print("Saved plot fig to visualization_tsit5.png")
            plt.clf(); plt.figure()               
                                    
        if self.args.sample_type == "srk":
            with torch.no_grad():
                y_shape = (config.sampling.batch_size, config.data.dimension)
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                
            scale = 50.0 if config.data.dataset == "Quadratic" \
                           else (2.0 / math.sqrt(2 * math.pi))

            y_0 = y_0 * scale
            y   = y   * scale

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(config.sampling.batch_size):
                ax[0].plot(x_0[i, :].cpu(), y_0[i, :].cpu())

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            n_tests = config.sampling.batch_size // 10

            for i in range(y.shape[0]):
                ax[1].plot(free_input[i, :].cpu(), y[i, :].cpu(), alpha=1)
            print('Calculate Confidence Interval:')
            power_res = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # power_res2 = calculate_ci(y, y00, n_tests=n_tests)
            # print(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            logging.info(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # logging.info(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            ax[1].set_title(f'resolution-free, power(avg of 30 trials): {power_res}')
            # ax[1].set_title(f'resfree 1: {power_res}, resfree 2: {power_res2}')
            # plt.savefig('result.png')
            # np.savez(args.log_path + '/rawdata', x_0=x_0.cpu().numpy(), y_0=y_0.cpu().numpy(), free_input=free_input.cpu().numpy(), y=y.cpu().numpy())

        else:
            y_0 = y_0.squeeze(-1)
            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W,  self.sde.eps, config.data.dataset)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            if config.data.dataset == 'Melbourne':
                lp = 10
                n_tests = y.shape[0] // 10
                y = data_inverse_scaler(y)
            if config.data.dataset == 'Gridwatch':
                lp = y.shape[0]
                n_tests = y.shape[0] // 10
                plt.ylim([-2, 3])

            for i in range(lp):
                ax[0].plot(x_0[i, :].cpu(), y[i, :].cpu())
                ax[1].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), c='black', alpha=1)

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            for i in range(lp):
                ax[1].plot(x_0[i, :].cpu(), y[i, :].cpu(), alpha=1)

            power = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: grid, 0th: {power}')

            ax[1].set_title(f'grid, power(avg of 30 trials):{power}')

        # Visualization figure save
        plt.savefig('visualization_default.png')
        print("Saved plot fig to {}".format('visualization_default.png'))
        plt.clf()
        plt.figure()