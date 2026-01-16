import torch
import sentencepiece as spm
from config.gpt_config import GPTConfig
from model.gpt_model import GPT

def sample(model, idx, max_length=600, temperature=1.0, top_k=50, top_p=0.95):
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    # 1文字ずつ順番に生成していくループ
    for _ in range(max_length - idx.size(1)):
        # 前後の文脈を見て、次の文字の確率（logits）を計算
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature # 一番最後の文字の予測だけを使う

        # Top-k フィルタリング
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")

        # Top-p サンプリング
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        
        # 修正箇所: インデックス指定のミスを修正
        sorted_indices_to_remove = cumulative_probs > top_p
        # 最初の1つは残す（全て消えるのを防ぐ）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits[sorted_indices_to_remove] = -float("inf")
        # 元の順番に並べ戻す
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        # 最終的な確率から、ランダムに1つ選ぶ
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 今までの文章に、新しく選んだ1文字を合体させる
        idx = torch.cat((idx, next_token), dim=1)

        # もし終了トークンがあれば、そこでループを抜ける処理を入れるとより良いです

    return idx

def main():
    cfg = GPTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg).to(device)
    
    # 重みのロード
    try:
        model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    except:
        print("Warning: Trained weights not found. Using random weights.")

    # トークナイザーの読み込み
    sp_model = spm.SentencePieceProcessor()
    sp_model.load("data/tokenizer/my_tokenizer.model") 

    # 入力文（プロンプト）
    prompt = "<user>こんにちは、最近どう？<bot>"
    input_ids = torch.tensor([sp_model.encode(prompt)], dtype=torch.long)

    # 生成開始
    print("Generating...")
    output_ids = sample(model, input_ids, max_length=100)
    response = sp_model.decode(output_ids[0].tolist())

    print("\nMemoryAIの応答:")
    print(response)

if __name__ == "__main__":
    main()
