# memoryAI - 長期記憶強化型GPTモデル
→[ライセンスとポリシー](https://github.com/RockHopperPenguin64/memoryAI/blob/main/jpLicense.md)

→[🌐English version here](https://github.com/RockHopperPenguin64/memoryAI/blob/main/README.md)

---

### 🥳カウントダウン!! =====>  _**🤔あと2日🤩**_

---

### 🙇‍ プロジェクトのステータスとライセンスに関する重要なお知らせとお詫び 🙇‍

本プロジェクトをご覧いただき、またクローンしていただきありがとうございます。

プロジェクトの非オープンソースライセンスおよび現在の開発状況に関する明確な情報が不足しており、ご迷惑をおかけしたことを心よりお詫び申し上げます。

**プロジェクトの進捗について：**  
長期記憶機能はまだ実装されていません。現在は開発ロードマップ上のトレーニング段階にあります。

**ライセンスについて：**  
本プロジェクトは**オープンソースではありません。**  
コードの閲覧・学習は歓迎しますが、著者の明示的な許可なくクローン、改変、再配布することはできません。

本ポリシーを知らずにリポジトリをクローンされた場合、他プロジェクトへのコードの利用はお控えください。

ご理解のほど、よろしくお願いいたします。今後はより透明性の高い情報発信に努めてまいります。

---

## 概要

**memoryAI**は、長期記憶機能を備えることを目指して設計されたカスタムGPTモデル（Transformerベース）です。  
過去の会話内容をSQLデータベースに保存し、そこから記憶を検索・参照することで、文脈を維持した対話を実現することを目指しています。

 

## 主な特徴

- **カスタムGPTアーキテクチャ**  
  PyTorch製。RMSNorm、MultiHeadAttention、位置埋め込み、深さ・幅の設定などを独自実装。

- **日本語対応**  
  トークナイザーにSentencePiece＋Juman++を採用し、日本語テキスト処理を最適化。

- **サンプリングと生成制御**  
  Top-k・Top-pフィルタリング、温度スケーリング、繰り返しペナルティ等でテキスト生成を制御。

- **記憶検索ロジック（予定）**  
  SQLベースの記憶システムで過去の対話をトークンレベルで照合・スコアリングし、最も適切な記憶をTop-k選出・再ランキングで抽出。

- **トレーニングパイプライン**  
  Mixed Precision Training（GradScaler・autocast）、Cosine Learning Rate Scheduler、Perplexity評価などを実装。

## 開発ロードマップ

- **フェーズ1: モデルコア完成**  ✅[完了]
  - GPTアーキテクチャ・推論ロジックの完成
  - 基本的なトレーニングパイプラインの実装

- **フェーズ2: トレーニング＆最適化** ⏳[進行中]
  - 日本語データセットによる学習開始
  - Mixed PrecisionやCosine LR Schedulerを用いたトレーニング最適化

- **フェーズ3: 記憶システム統合** 🔜[予定]
  - SQLベースの記憶検索ロジック設計・実装
  - 記憶システムの統合で過去会話参照を実現

- **フェーズ4: 文脈対話の実現** 🚀[今後]
  - 長期記憶を活用した文脈保有型応答の最終実装
 

## プロジェクト構成
model/

└── **gpt_model.py**    _# Transformerモデル本体_

config/

└── **gpt_config.py**   _# dataclassによるモデル設定_

infrence/

├── **generate.py**     _# テキスト生成ロジック_

└── **sampling.py**     _# サンプリング・フィルタ処理_

train/

├── **train.py**       _ # Optimizer・Schedulerを用いた学習ループ _

├── **evaluate.py**     _# Perplexity等の評価指標_

└── **dataset.py**      _# データセット読み込み・前処理_

utils/ 

└── **lr_scheduler.py** _# Cosine LR Scheduler_

 

## 使用技術

- Python / PyTorch  
- SentencePiece / Juman++  
- SQL（記憶ストレージ・検索：予定）  
- Mixed Precision Training  
- Cosine Learning Rate Scheduler
