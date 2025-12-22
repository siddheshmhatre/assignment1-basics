import argparse
import numpy as np
import torch
import wandb
import os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--it", type=int, default=1000)
    parser.add_argument("--max_learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_learning_rate", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="cs336_assignment_1", config=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), args.lr, args.weight_decay, args.betas, args.eps)
    
    train_ds = np.memmap(args.train_path, dtype=np.uint16, mode='r')
    val_ds = np.memmap(args.val_path, dtype=np.uint16, mode='r')

    step = 0
    for epoch in range(args.num_epochs):
        for _ in range(args.it // args.num_epochs):
            lr = get_cosine_learning_rate(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            model.train()
            x, y = get_batch(train_ds, args.batch_size, args.context_length, device)
            
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), args.max_l2_norm)
            optimizer.step()

            if args.wandb:
                wandb.log({"train_loss": loss.item(), "lr": lr, "step": step})

            if step > 0 and step % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch(val_ds, args.batch_size, args.context_length, device)
                    v_logits = model(vx)
                    v_loss = loss_fn(v_logits.view(-1, v_logits.size(-1)), vy.view(-1))
                    if args.wandb:
                        wandb.log({"val_loss": v_loss.item(), "step": step})
                
                save_checkpoint(model, optimizer, step, os.path.join(args.checkpoint_path, f"ckpt_{step}.pt"))
            
            step += 1

if __name__ == "__main__":
    main()