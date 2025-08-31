import torch
import torch.nn.functional as F

def sample_logits(logits, temperature=1.0, top_k=50, top_p=0.95)
  #温度調整
  logits = logits / temperature

  #top-kフィルタリング
  if top_k > 0:
    values, _ = torch.topk(logits, top_k)
    logits[logits < values[:, [-1]]] = -float("Inf")

  #top-pフィルタリング
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = F.softmax(sorted_logits, dim=1).cumsum(dim=-1)
  sorted_indices_to_remove = cumulative_probs > top_p
  sorted_logits[sorted_indices_to_remove] = -float("Inf")
  logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

  #サンプリング
  probs = F.softmax(logits, dim=-1)
  next_token = torch.multinomial(probs,num_samples=1)
  return next_token
