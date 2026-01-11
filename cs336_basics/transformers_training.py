import os
import math
import random
import torch
import numpy as np
import torch.nn as nn
import numpy.typing as npt

from typing import IO, BinaryIO
from typing import Iterable

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        row_idxs = torch.arange(logits.shape[0])
        loss = -logits[row_idxs, targets] + logits.exp().sum(dim=-1).log()
        return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]

                # Initialize state if this is the first step for this parameter
                if len(state) == 0:
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                    state["t"] = 0

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]
                
                grad = param.grad

                param.add_(param, alpha=-lr * weight_decay)

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha = lr * (math.sqrt(bias_correction2) / bias_correction1)

                denom = v.sqrt().add_(eps)
                param.addcdiv_(m, denom, value=-alpha)

def get_cosine_learning_rate(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters ) * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        frac = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cos = 1 + math.cos( frac * math.pi)
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * cos
        return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grad_norm = 0
    for param in parameters:
        grad = param.grad
        if grad is not None:
            grad_norm += (torch.linalg.norm(grad.data))**2

    grad_norm = torch.sqrt(grad_norm)

    if grad_norm > max_l2_norm:
        scale = max_l2_norm / (grad_norm + 10e-6)
        for param in parameters:
            grad = param.grad
            if grad is not None:
                param.grad.data = param.grad.data * scale

    return parameters

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = [random.randint(0, dataset.shape[0] - context_length - 1) for _ in range(batch_size)]

    inp = np.zeros((batch_size, context_length))
    labels = np.zeros((batch_size, context_length))

    for b_idx, start_idx in enumerate(start_indices):
        inp[b_idx] = dataset[start_idx: start_idx + context_length]
        labels[b_idx] = dataset[start_idx + 1: start_idx + context_length + 1]

    inp = torch.from_numpy(inp).long().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    return (inp, labels)

def save_checkpoint( model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    state_dict = {}
    state_dict['model'] = model.state_dict()
    state_dict['optim'] = optimizer.state_dict()
    state_dict['t'] = iteration
    torch.save(state_dict, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optim'])
    return state_dict['t']
