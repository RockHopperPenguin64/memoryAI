# memoryAI - Long-Term Memory Enhanced GPT Model
→[License&Policy](https://github.com/RockHopperPenguin64/memoryAI/blob/main/License.md)

→[🇯🇵日本語はこちら](https://github.com/RockHopperPenguin64/memoryAI/blob/main/jpREADME.md)

## Overview

**memoryAI** is a custom-built Transformer-based GPT model designed to support long-term memory capabilities.  
The project aims to enable contextual dialogue by integrating a memory retrieval system that references past conversations stored in a SQL database.

 

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

## Development Roadmap

- **Phase 1: Model Core Completion**  ✅[Completed]
  - Finalize the GPT architecture and inference logic.
  - Implement a basic training pipeline.

- **Phase 2: Training & Optimization** ⏳[In Progress]
  - Begin model training with Japanese datasets.
  - Optimize the training process using mixed precision and a cosine learning rate scheduler.

- **Phase 3: Memory System Integration** 🔜[Planned]
  - Design and implement the SQL-based memory retrieval logic.
  - Integrate the memory system to allow the model to reference past conversations.

- **Phase 4: Contextual Dialogue** 🚀[Future]
  - Finalize the overall system to enable long-term memory-aware responses.
 

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

 

## Technologies Used

- Python / PyTorch  
- SentencePiece / Juman++  
- SQL (planned for memory storage and retrieval)  
- Mixed Precision Training  
- Cosine Learning Rate Scheduler

