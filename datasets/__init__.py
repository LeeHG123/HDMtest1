"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""

import numpy as np
import torch

from datasets.quadratic import QuadraticDataset
from datasets.gaussian  import GaussianDataset 
from datasets.gaussian2d import Gaussian2DDataset

def data_scaler(data):
    return data * 2. - 1.


def data_inverse_scaler(data):
    return (data + 1.) / 2.

def get_dataset(config):
    if config.data.dataset == "Quadratic":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42)
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43)
    elif config.data.dataset == "Gaussian":
        dataset = GaussianDataset(num_data=config.data.num_data,
                                  num_points=config.data.dimension,
                                  seed=42)
        test_dataset = GaussianDataset(num_data=config.data.num_data,
                                       num_points=config.data.dimension,
                                       seed=43)
    elif config.data.dataset == "Gaussian2D":
        dataset = Gaussian2DDataset(num_data=config.data.num_data,
                                    num_points=config.data.dimension,
                                    seed=42)
        test_dataset = Gaussian2DDataset(num_data=config.data.num_data,
                                         num_points=config.data.dimension,
                                         seed=43)        
    else:
        raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")
    return dataset, test_dataset


