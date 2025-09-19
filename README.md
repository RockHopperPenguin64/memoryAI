# memoryAI - Long-Term Memory Enhanced GPT Model

## Overview

**memoryAI** is a custom-built Transformer-based GPT model designed to support long-term memory capabilities.  
The project aims to enable contextual dialogue by integrating a memory retrieval system that references past conversations stored in a SQL database.

---

## Key Features

- **Custom GPT Architecture**  
  Built from scratch using PyTorch, including RMSNorm, MultiHeadAttention, Positional Embedding, and configurable depth and width.

- **Japanese Language Support**  
  Tokenization is handled via SentencePiece and Juman++, optimized for Japanese text processing.

- **Sampling and Inference Control**  
  Implements top-k and top-p filtering, temperature scaling, and repetition penalty for controlled text generation.

- **Memory Retrieval Logic (Planned)**  
  A SQL-based memory system will allow the model to retrieve relevant past dialogue based on token-level matching and weighted scoring.  
  Top-k filtering and re-ranking will be used to select the most contextually appropriate memory.

- **Training Pipeline**  
  Includes mixed precision training (GradScaler, autocast), cosine learning rate scheduling, and perplexity-based evaluation.

---

## Project Structure
model/

└── **gpt_model.py**    _# Transformer model implementation_

config/

└── **gpt_config.py**   _# Model configuration via dataclass_

infrence/

├── **generate.py**     _# Text generation logic_

└── **sampling.py**     _# Sampling and filtering functions_

train/

├── **train.py**       _ # Training loop with optimizer and scheduler _

├── **evaluate.py**     _# Evaluation metrics (e.g., perplexity) _

└── **dataset.py**      _# Dataset loader and preprocessing_

utils/ 

└── **lr_scheduler.py** _# Cosine learning rate scheduler_

---

## Technologies Used

- Python / PyTorch  
- SentencePiece / Juman++  
- SQL (planned for memory storage and retrieval)  
- Mixed Precision Training  
- Cosine Learning Rate Scheduler

---

## Development Roadmap

1. **Model Completion** – Finalize GPT architecture and inference logic  
2. **Training Phase** – Begin model training with curated Japanese datasets  
3. **Memory System Integration** – Implement SQL-based memory retrieval logic  
4. **Contextual Dialogue** – Enable long-term memory-aware responses

---

## License & Usage Policy

This project is **not licensed under any open-source license**.  
You are welcome to **view and learn from the code**, but **cloning, modifying, redistributing, or deploying any part of this repository** is **not permitted** without explicit permission from the author.

Additionally, **unauthorized access to any linked services, including SQL servers or APIs, is strictly prohibited**.

If you wish to use or adapt this code, please contact me via GitHub Discussion or DM.

**e-mail** : fankymonkey876@gmail.com
