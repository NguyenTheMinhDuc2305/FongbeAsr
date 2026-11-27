"""Native PyTorch equivalents of the six k2 Swoosh operations used upstream."""
import random
import numpy as np
import torch
import torch.nn.functional as F


def _forward(x, offset, constant):
    # Evaluate nonlinearities in float32 under mixed precision, then restore dtype.
    z = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (F.softplus(z - offset) - 0.08 * z - constant).to(x.dtype)


def _both(x, offset, constant):
    z = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return _forward(x, offset, constant), (torch.sigmoid(z - offset) - 0.08).to(x.dtype)


def swoosh_l(x):
    return _forward(x, 4.0, 0.035)


def swoosh_r(x):
    return _forward(x, 1.0, 0.313261687)


swoosh_l_forward = swoosh_l
swoosh_r_forward = swoosh_r


def swoosh_l_forward_and_deriv(x):
    return _both(x, 4.0, 0.035)


def swoosh_r_forward_and_deriv(x):
    return _both(x, 1.0, 0.313261687)


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
