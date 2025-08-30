import torch
from torch.utils.data import Dataset
import json

class RyoDataset(Dataset):
  def __init__(self, path, seq_len=600)
    self.data = []
    self.seq_len = seq_len

    with open(path, "r", encoding = "utf-8") as f:
      for line in f:
        item = json.loads(line)
        tokens = item["tokens"]
        if len(tokens) >= 2: #最低限の長さの確保
          self.data.append(tokens:[seq_len])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    tokens = self.data[idx]
    x = torch.tensor(tokens, dtype=torch.long)
    y = torch.tensor(tokens, dtype=torch.long)

    return x, y
          
