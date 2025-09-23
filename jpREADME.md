# memoryAI - 長期記憶を強化したGPTモデル  
[→ライセンスと利用規約](https://github.com/RockHopperPenguin64/memoryAI/blob/main/jpLicense.md)

## 概要

**memoryAI** は、長期記憶機能を備えたカスタム構築のTransformerベースGPTモデルです。  
このプロジェクトは、SQLデータベースに保存された過去の会話を参照する記憶検索システムを統合することで、文脈に応じた対話を可能にすることを目的としています。

## 主な特徴

- **カスタムGPTアーキテクチャ**  
  PyTorchを用いてゼロから構築。RMSNorm、MultiHeadAttention、位置埋め込み、深さと幅の設定が可能。

- **日本語対応**  
  SentencePieceとJuman++によるトークナイズで、日本語テキスト処理に最適化。

- **サンプリングと推論制御**  
  top-k・top-pフィルタリング、温度スケーリング、繰り返しペナルティを実装し、生成テキストの制御が可能。

- **SQLベースの記憶**  
  SQLベースの記憶システムにすることで、SQLの強みを最大限に活用

- **学習パイプライン**  
  混合精度学習、コサイン学習率スケジューリング、パープレキシティによる評価を含む。

## プロジェクト構成
→[フォルダ構成](https://github.com/RockHopperPenguin64/memoryAI/blob/main/jpForder.txt)


## 使用技術

- Python / PyTorch  
- SentencePiece / Juman++  
- SQL（記憶保存と検索に使用予定）  
- 混合精度学習  
- コサイン学習率スケジューラ

## 開発ロードマップ

1. **モデル完成** – GPTアーキテクチャと推論ロジックの最終化  
2. **学習フェーズ** – 日本語データセットによるモデル学習開始  
3. **記憶システム統合** – SQLベースの記憶検索ロジックの実装  
4. **文脈対話対応** – 長期記憶に基づく応答の実現
5. 
