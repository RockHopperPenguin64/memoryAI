import torch
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from config.gpt_config import GPTConfig
from model.gpt_model import GPT
from train.dataset import MyDataset # 名前に注意
from utils.lr_scheduler import cosine_lr # フォルダ名に注意

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, accum_steps=8, clip=1.0):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # 混合精度で計算
        with autocast():
            _, loss = model(x, y)
            loss = loss / accum_steps # 蓄積のために割る

        # 逆伝播（間違いの分析）
        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            # 勾配クリッピング（学習が爆発するのを防ぐ）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # お利口になるための更新（反省）
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)

# evaluate関数はさっきと同じなので省略

def main():
    # 準備
    cfg = GPTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存先フォルダを作っておく
    os.makedirs("checkpoints", exist_ok=True)
    
    model = GPT(cfg).to(device)

    # データ読み込み
    train_ds = MyDataset("data/processed/train.jsonl", seq_len=600)
    valid_ds = MyDataset("data/processed/valid.jsonl", seq_len=600)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=2, shuffle=False)

    # 最適化の設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = cosine_lr(optimizer, base_lr=3e-4, warmup=100, total_steps=1000) # warmup等はデータ量に合わせる
    scaler = GradScaler()

    print(f"Starting training on {device}...")

    # 学習ループ（とりあえず2周）
    for epoch in range(2):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device)
        val_ppl = evaluate(model, valid_loader, device) # evaluate.pyの関数
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val PPL = {val_ppl:.2f}")
        torch.save(model.state_dict(), f"checkpoints/ckpt_ep{epoch}.pt")

if __name__ == "__main__":
    main()
