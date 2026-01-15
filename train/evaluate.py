import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
# dataset.pyでクラス名をMyDatasetにした場合はこちら
from train.dataset import MyDataset 
from config.gpt_config import GPTConfig
from model.gpt_model import GPT
import os

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast():
            # モデルにデータ(x, y)を入れて、損失(loss)を受け取る
            _, loss = model(x, y)
        
        # loss.item() は1トークンあたりの平均なので、全部のトークン数を掛けて合計を出す
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    if total_tokens == 0: return float('inf') # データがない場合

    # 平均損失からPerplexityを計算
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

def main():
    cfg = GPTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの作成
    model = GPT(cfg).to(device)
    
    # 重みファイルの読み込み（ファイルがあるか確認してから）
    ckpt_path = "checkpoints/best.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("Warning: No checkpoint found. Evaluating an untrained model.")

    # データの読み込み
    # ここもMyDatasetに合わせる
    try:
        valid_ds = MyDataset("data/processed/valid.jsonl", seq_len=cfg.max_seq_len)
        valid_loader = DataLoader(valid_ds, batch_size=2)

        # 評価実行
        ppl = evaluate(model, valid_loader, device)
        print(f"Validation Perplexity: {ppl:.2f}")
    except FileNotFoundError:
        print("Error: Valid data file not found.")

if __name__ == "__main__":
    main()
