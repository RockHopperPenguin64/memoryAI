from dataclasses import dataclass

@dataclass
class GPTConfig:
    # モデル構造
    vocab_size: int = 50000         # SentencePieceの語彙サイズ（32k or 50k）
    n_layers: int = 16              # Transformer層数
    n_heads: int = 32               # Attentionヘッド数
    d_model: int = 1024             # 埋め込み次元
    d_ff: int = 3072                # FFNの隠れ層次元
    dropout: float = 0.17           # Dropout率
    max_seq_len: int = 1024         # 最大シーケンス長（位置埋め込み）

    # 特殊トークン（必要に応じて）
    bos_token_id: int = 0
    eos_token_id: int = 1
    user_token_id: int = 2
    bot_token_id: int = 3
    pad_token_id: int = 4

    # モデル運用設定
    tie_embeddings: bool = True     # 入力埋め込みと出力埋め込みを共有
    use_checkpoint: bool = True     # 勾配チェックポイントを有効化（メモリ節約）
