import torch

def cosine_lr(optimizer, base_lr, warmup, total_steps):
    def lr_lambda(step):
        # 1. Warmup
        if step < warmup:
            return step / warmup
        
        # 進捗計算（1を超えないようにガード）
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        # progress=0 のとき 1.0, progress=1 のとき 0.5 になる設定
        return 0.5 * (1 + (1 - progress)**2)
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
