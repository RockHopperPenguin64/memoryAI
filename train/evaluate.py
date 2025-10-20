import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from train.dataset import Dataset
from config.gpt_config import GPTConfig
from model.gpt_model import GPT

@torch.no_grad()
def evaluate(model, loader, device):
  model.eval()
  total_loss, total_tokens = 0.0, 0

  for x, y in loader:
    x, y = x.to(device), y.to(device)
    with autocast():
      _, loss = model(x, y)
    total_loss += loss.item() * x.numel()
    total_tokens += x.numel()
  ppl = torch.exp(torch.tensor(total_loss / total_tokens))
  return ppl.item()

def main():
  cfg = GPTConfig()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = GPT(cfg).to(device)
  model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))

  #データの読み込み
  vaild_ds = Dataset("data/processed/valid.jsonl", seq_len=cfg.max_seq_len)
  valid_loader = DataLoader(valid_ds, batch_size=2)

  #評価実行
  ppl = evaluate(model, valid_loader, device)
  print(f"Validation Perplexity: {ppl:.2f}")

if __name__ == "__main__":
  main()
