def cosine_lr(optimizer, base_lr, warmup, total_steps):
  def lr_lambda(step):
    if step < warmup:
      return step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return 0.5 * (1 + (1 - progress)**2) #緩やかなCosine減衰
    
return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
