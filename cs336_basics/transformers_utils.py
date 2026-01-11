import torch
import random
import numpy as np
import torch.nn as nn

from typing import List, Dict
from collections import defaultdict
from cs336_basics.transformers_arch import SoftMax

def decode(model: nn.Module, prompt_token_ids: torch.Tensor, eos_token_id: int, max_num_tokens: int, temperature: float, top_p: float = None) -> List[int]:
    next_token_idx = None
    softmax = SoftMax()
    num_prompt_tokens = prompt_token_ids.shape[1]
    generated_tokens = torch.zeros(1, max_num_tokens, dtype=prompt_token_ids.dtype, device=prompt_token_ids.device) # Assuming batch_size of 1 for now for decoding
    generated_tokens[0, :num_prompt_tokens] = prompt_token_ids
    generated_tokens

    while next_token_idx != eos_token_id and num_prompt_tokens < max_num_tokens:
        output = model(generated_tokens)[:, num_prompt_tokens - 1, :]
        output = output / ( temperature + 1e-6 )
        output = softmax(output, dim=-1)

        if top_p is not None:
            next_token_idx = nucleus_sampling(output, top_p)
        else:
            next_token_idx = torch.multinomial(output, num_samples=1)

        generated_tokens[0, num_prompt_tokens] = next_token_idx

        num_prompt_tokens += 1

    generated_tokens = generated_tokens[0, :num_prompt_tokens].cpu().numpy().tolist()
    return generated_tokens

def nucleus_sampling(logits: torch.Tensor, top_p: float) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_sum_logits = torch.cumsum(sorted_logits, dim=-1)
    threshold_mask = cum_sum_logits <= top_p
    threshold_mask[0, 1:] = threshold_mask[0, :-1].clone()
    threshold_mask[0][0] = True
    sorted_logits = sorted_logits[threshold_mask]
    sorted_indices = sorted_indices[threshold_mask]
    normalized_logits = sorted_logits / ( sorted_logits.sum() + 1e-6 )
    next_token_idx = torch.multinomial(normalized_logits, 1)
    return sorted_indices[next_token_idx]

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    class MockModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            
        def forward(self, x):
            # Maps (1, seq_len) -> (1, seq_len, vocab_size)
            batch, seq_len = x.shape
            return torch.randn(batch, seq_len, self.vocab_size)

    # 1. Setup Parameters
    vocab_size = 100
    seq_len = 200
    max_num_tokens = 256
    temperature = 5.0
    top_p = 0.95
    eos_token_id = 0

    model = MockModel(vocab_size)
    prompt_token_ids = torch.randint(0, vocab_size, (1, seq_len))
    
    out = decode(
        model=model,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=eos_token_id,
        max_num_tokens=max_num_tokens,
        temperature=temperature,
        top_p=top_p
    )

    out = decode(
        model=model,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=eos_token_id,
        max_num_tokens=max_num_tokens,
        temperature=temperature,
        top_p=None
    )