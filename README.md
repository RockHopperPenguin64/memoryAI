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


---

## License & Usage Policy

This project is **not licensed under any open-source license**.  
You are welcome to **view and learn from the code**, but **cloning, modifying, redistributing, or deploying any part of this repository** is **not permitted** without explicit permission from the author.

Additionally, **unauthorized access to any linked services, including SQL servers or APIs, is strictly prohibited**.

If you wish to use or adapt this code, please contact me via GitHub Discussion or DM.

**e-mail** : fankymonkey876@gmail.com
