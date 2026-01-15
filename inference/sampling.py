import torch
import torch.nn.functional as F

def sample_logits(logits, temperature=1.0, top_k=50, top_p=0.95):
    # 1. 温度調整（Temperature）
    # 1.0より高いと多様に、低いと堅実になります
    logits = logits / max(temperature, 1e-5) # 0除算防止

    # 2. Top-k フィルタリング
    if top_k > 0:
        # 上位k個の値を抽出
        values, _ = torch.topk(logits, top_k)
        # k番目の値より小さいものは-inf（確率ゼロ）にする
        logits[logits < values[:, [-1]]] = -float("inf")

    # 3. Top-p (Nucleus) フィルタリング
    # 累積確率がpを超えるものを除外
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # 累積確率がpを超えたインデックスを特定
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # [重要] 少なくとも1つの候補は残るようにシフトする（安全策）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 該当するロジットを-infに
    sorted_logits[sorted_indices_to_remove] = -float("inf")
    
    # 元の順番に並べ直す
    logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    # 4. 最終的なサンプリング
    probs = F.softmax(logits, dim=-1)
    # 確率分布に従ってランダムに1つ選択
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
