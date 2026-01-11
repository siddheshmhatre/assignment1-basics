import time
import argparse
import numpy as np
import torch
import wandb
import os

from tqdm import tqdm
from cs336_basics.transformers_arch import TransformerLM
from cs336_basics.transformers_training import (
    CrossEntropyLoss,
    AdamW,
    get_cosine_learning_rate,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)
from cs336_basics.transformers_utils import decode, set_seed
from cs336_basics.tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=2e-8)
    parser.add_argument("--max_learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)
    parser.add_argument("--max_l2_norm", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=10**9)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    if args.min_learning_rate is None:
        args.min_learning_rate = 0.1 * args.max_learning_rate

    tokens_per_iter = args.batch_size * args.context_length
    total_iters = args.num_tokens // tokens_per_iter

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = total_iters

    mode = "offline" if args.wandb_offline else "online"
    wandb.init(project="cs336_assignment_1", config=args, mode=mode)
    folder_name = f"{wandb.run.name}_{wandb.run.id}"
    checkpoint_dir = os.path.join(args.checkpoint_path, folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), weight_decay=args.weight_decay, betas=args.betas, eps=args.eps)
    
    train_ds = np.load(args.train_path, mmap_mode="r")
    val_ds = np.load(args.val_path, mmap_mode="r")

    if args.overfit:
        total_iters = 10_000
        args.batch_size = 2
        args.eval_freq = 1_000
        x, y = get_batch(train_ds, args.batch_size, args.context_length, device)

    step = 0
    progress_bar = tqdm(range(total_iters), desc="Training", dynamic_ncols=True)
    
    for step in progress_bar:
        lr = get_cosine_learning_rate(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()

        if not args.overfit:
            x, y = get_batch(train_ds, args.batch_size, args.context_length, device)
        
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        wandb.log({**model.get_per_block_gradient_norms(), "step" : step})

        gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.1e}"})

        wandb.log({"train_loss": loss.item(), "lr": lr, "step": step})

        if step > 0 and step % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                if not args.overfit:
                    x, y = get_batch(val_ds, args.batch_size, args.context_length, device)
                v_logits = model(x)
                v_loss = loss_fn(v_logits.view(-1, v_logits.size(-1)), y.view(-1))
                wandb.log({"val_loss": v_loss.item(), "step": step})
            
        step += 1

	# Only save checkpoint at the end
    save_checkpoint(model, optimizer, step, os.path.join(checkpoint_dir, f"ckpt_{step}.pt"))

    tokenizer = Tokenizer.from_files("data/TinyStoriesV2-GPT4-train.pkl", "", ["<|endoftext|>"])
    sample_tokens = tokenizer.encode("Once upon a time")
    sample_tokens = torch.Tensor(sample_tokens).long().to(device)
    output = decode(model, sample_tokens.unsqueeze_(0), 0, 256, temperature=1.0, top_p=0.95)
    print(tokenizer.decode(output))

if __name__ == "__main__":
    main()
