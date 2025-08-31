import torch
from config.gpt_config import GPTConfig
from model.gpt_model import GPT
from train.dataset import RyoDataset
import sentencepiece as spm

def sample(model, idx, max_length=600, temprature=1.0, top_k=50, top_p=0.95):
  model.eval()
  device = next(model.parameters()).device
  idx = idx.to(device)

  for _ in range(max_length - idx.size(1)):
    logits, _ = model(idx)
    logits = logits[:, -1, :] / temperature

    #Top-k filtering
    if top_k > 0:
      values, _ = torch.topk(logits, top_k)
      logits[logits < values[:, [-1]]] = -float("Inf")

    #Top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_indices_to_remoce = cumulative_probs > top_p
    sorted_logits[sorted_indices_to_remove = -float("Inf")
    logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_sample=1)
    idx = torch.cat((idx, next_token), dim=1)

  return idx

def decode(tokens, sp_model):
  return sp_model.decode_ids(tokens.tolist())

def main():
  cfg = GPTConfig(max_seq_len=2048)
  devicce = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = GPT(cfg).to(device)
  model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))

  #トークナイザーの読み込み
  sp_model = spm.SentencePieceProcessor()
  sp_model.load("data/tokenizer/ryo_tok.model")

  #入力文
  prompt = "<user>こんにちは、最近どう？<bot>"
  input_ids = torch.tensor([sp_model.encode(prompt)], dtype=torch.long)

  #生成
  output_ids = sample(model, input_ids, max_length=600)
  response = decode(output_ids[0], sp_model)

  print("MemoryAIの応答")
  print(response)

if __name__ == "__main__":
  main()
