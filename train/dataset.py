import torch
from torch.utils.data import Dataset
import json

class MyDataset(Dataset): # クラス名がDatasetだと紛らわしいので少し変えました
    def __init__(self, path, seq_len=600): # : を追加
        self.data = []
        self.seq_len = seq_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    tokens = item["tokens"]
                    
                    if len(tokens) >= 2:
                        chunk = tokens[:self.seq_len]
                        self.data.append(chunk)
                except Exception as e:
                    print(f"エラーの行を飛ばしたよ: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # PyTorchが計算できるように「テンソル」という形に変える
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens, dtype=torch.long)

        return x, y
