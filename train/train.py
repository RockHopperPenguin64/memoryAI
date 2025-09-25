import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from config.gpt_config import GPTConfig
from model.gpt_model import GPT
from train.dataset import Dataset
from utils.lr.lr_scheduler import cosine_lr

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, accum_steps=8, clip=0.3):
  model.train(
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
      x, y = x.to(device), y.to(device)

      with autocast():
        _, loss = model(x, y)
        loss = loss / accum_steps

    scaler.scale(loss).backward()
    if (step + 1) % accum_steps == 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad(set_to_no=True)
      scheduler.step()

    total_loss += loss.item() * accum_steps

return total_loss / len(loader)

@torch.no_grad()
def evalute(model, loader, device):
  model.eval()
  total_loss, total_tokens = 0.0, 0

  for x, y in loader:
    x, y = x.to(device), y.to(device)
    with autocast():
      /, loss = model(x, y)
    total_loss += loss.item() * x.numel()
    total_tokens += x.numel()

  ppl = torch.exp(torch.tensor(total_loss / total_tokens))
  return ppl.item()

def main():
  #設定
  cfg = GPTConfig()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = GPT(cfg).to(device)

  #データ読み込み（トークナイズ済み）
  train_ds = Dataset("data/processed/train.jsonl", seq_len=600)
  valid_ds = Dataset("data/processed/valid.jsonl", seq_len=600)
  train_loader = DataLoader(train_ds, batch_size = 2, shuffle = True)
  valid_loader = DataLoader(valid_ds, batch_size = 3,

  #最適化
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
  scheduler = cosine_lr(optimizer, base_lr=3e-4, warmup=4000, total_steps=10000)
  scaler = GradScaler()

  #学習ループ
  for epoch in range(2):
    train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device)
    var_ppl = evaluate(model, valid_loader, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val PPL = {val_ppl:.2f}")
    torch.save(model.state_dict(),f"checkpoints/ckpt_ep{epoch}.pt")

if __name__ == "__name__":
  main()
