# Amazon Comment Summarization Project

A comprehensive framework for generating and training models on multi-level review summarization using the Amazon 2023 dataset.

## Overview

ETSP (E-commerce Text Summarization Project) is designed to create high-quality training data for review summarization models across different complexity levels. The project includes:

- **Automated data generation** from Amazon reviews (10 categories)
- **Multi-level summarization**:  Simple (A1), Complex (C1), and Hallucinated variants
- **Model fine-tuning** using Qwen language models with LoRA/QLoRA
- **DPO (Direct Preference Optimization)** support for preference learning

## Key Features

### 1. Review Data Generation (review_gen.ipynb)
- Downloads and processes Amazon 2023 review datasets across 10 product categories
- Generates 200 samples per category (2,000 total)
- Creates three summarization variants per product:
  - **Simple (A1)**: Beginner-friendly paragraph summaries (CEFR A1 level)
  - **Complex (C1)**: Professional bullet-point analysis with sentiment markers
  - **Hallucinated**: Complex summaries with intentional factual errors (for negative training)

#### Supported Categories: 
- Electronics
- Books
- Home & Kitchen
- Beauty & Personal Care
- Clothing, Shoes & Jewelry
- Toys & Games
- Sports & Outdoors
- Pet Supplies
- Automotive
- Office Products

### 2. Model Training (train_qwen.ipynb)
- Fine-tunes Qwen-2-1.5B model using QLoRA (4-bit quantization)
- Supports both SFT (Supervised Fine-Tuning) and DPO training
- Memory-efficient training for Google Colab environments
- Custom formatting for multi-turn conversations

## Installation

### Prerequisites
```bash
pip install transformers datasets peft bitsandbytes trl
pip install emoji langdetect
```

### Google Colab Setup
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### Data Generation

1. **Configure API credentials** in `review_gen.ipynb`:
```python
API_KEY = "your-dashscope-api-key"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
```

2. **Run generation**:
```python
SAMPLES_PER_CATEGORY = 200
MIN_REVIEWS_PER_PRODUCT = 5
MAX_REVIEWS_INPUT = 30
```

3. **Output**: JSONL files with structure: 
```json
{
  "asin": "B0047AQI4Q",
  "category": "Electronics",
  "reviews": [... ],
  "simple": "Most people like this TV.. .",
  "complex": "(+) Consistently praised...\n(-) Audio performance...",
  "hallucinated": "(+) Picture quality...[with subtle error]"
}
```

### Model Training

1. **Prepare dataset** from generated JSONL files

2. **Configure training parameters**:
```python
MODEL_NAME = "Qwen/Qwen2-1.5B"
TRAIN_FILE = "path/to/training_data_v2.jsonl"
OUTPUT_DIR = "./qwen-review-summarizer"
```

3. **Launch training**:
```python
# SFT training
trainer.train()

# DPO training (requires chosen/rejected pairs)
dpo_trainer.train()
```

## Data Format

### Input Reviews
- Filtered by length (20-1500 characters)
- Maximum 30 reviews per product
- HTML tags removed, whitespace normalized

### Summary Formats

**Simple (A1)**:
- 3-4 short sentences
- Basic vocabulary, present tense
- No bullet points

**Complex (C1)**:
- 3-6 bullet points with sentiment markers:
  - `(+)` Consensus strengths
  - `(-)` Consensus weaknesses
  - `(~)` Mixed/controversial opinions
- Sophisticated vocabulary
- 18-30 words per point

**Hallucinated**:
- Same structure as Complex
- Contains exactly ONE subtle factual error:
  - Attribute error (non-existent feature)
  - Quantity error (wrong numbers)
  - Sentiment error (reversed opinion)
  - Comparison error (false claim)

## Model Architecture

- **Base Model**: Qwen2-1.5B (2B parameters)
- **Quantization**: 4-bit NF4 via BitsAndBytes
- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 32
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Dropout: 0.05

## Training Details

- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4 (cosine schedule)
- **Warmup**: 100 steps
- **Max Sequence Length**: 2048 tokens
- **Precision**: FP16 mixed precision

## Project Structure

```
etsp/
├── review_gen.ipynb          # Data generation pipeline
├── train_qwen.ipynb          # Model training pipeline
├── ETSP/                     # Data storage (Google Drive)
│   ├── Electronics/
│   │   ├── Electronics. jsonl
│   │   ├── meta_Electronics.jsonl
│   │   └── Electronics_v2.jsonl
│   ├── Books/
│   └── ...
└── outputs/
    └── qwen-review-summarizer/  # Trained model checkpoints
```

## Data Sources

- **Reviews**: [Amazon Reviews 2023](https://mcauleylab.ucsd. edu/public_datasets/data/amazon_2023/raw/review_categories/)
- **Metadata**: [Amazon Metadata 2023](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/)

## Configuration Options

### Generation Parameters
- `temperature`: 0.3 (Simple/Complex), 0.5 (Hallucinated)
- `max_tokens`: 150 (Simple), 300 (Complex/Hallucinated)
- Retry logic: 3 attempts with exponential backoff

### Quality Validation
- Minimum 10 words per summary
- Simple: No bullet points allowed
- Complex/Hallucinated: Must contain sentiment markers
- Rejection phrase filtering

## Citation

If you use this project or dataset, please cite: 

```bibtex
@software{etsp2025,
  title={ETSP: E-commerce Text Summarization Project},
  author={u826318},
  year={2025},
  url={https://github.com/u826318/etsp}
}
```

## License

This project is provided for research and educational purposes.  Amazon review data is subject to Amazon's terms of service.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. 

## Contact

For questions or collaboration, please open an issue in this repository.

---

**Note**:  Requires GPU for model training (Google Colab L4 or equivalent recommended). Data generation uses Alibaba DashScope API (Qwen-Plus model).
