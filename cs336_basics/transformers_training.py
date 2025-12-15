import math
import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, logits: torch.Tensor, targets: torch.Tensor):
		logits = logits - logits.max(dim=-1, keepdim=True)[0]
		row_idxs = torch.arange(logits.shape[0])
		loss = -logits[row_idxs, targets] + logits.exp().sum(dim=-1).log()
		return loss.mean()

class AdamW(torch.optim.Optimizer):
	def __init__(self, params: dict[str, torch.Tensor], lr: float = 1e-3, weight_decay: float = 0.01, betas: tuple=(0.9, 0.999), eps: float = 1e-8):
		defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
		super().__init__(params, defaults)

	def step(self):
		for group in self.param_groups:
			lr = group["lr"]
			weight_decay = group["weight_decay"]
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			for p in group["params"]:
				if p.grad.data is None:
					continue
				state = self.state["p"]
				grad = p.grad.data
				m = state.get("m", torch.zeros_like(p))
				v = state.get("v", torch.zeros_like(p))
				t = state.get("t", 0)

				m = beta1 * m + (1 - beta1) * grad
				v = beta2 * v + (1 - beta2) * grad**2
				pow = t + 1
				alpha = lr * ( math.sqrt(1 - beta2**pow) / (1 - beta1**pow) )
				p.data -= alpha * ( m / (torch.sqrt(v) + eps) )
				p.data -= lr * weight_decay * p.data

				state["t"] = t + 1
				state["m"] = m
				state["v"] = v
