# Multi-Level Amazon Review Summarization with SFT and DPO

A two-stage training pipeline for generating readability-adapted Amazon review summaries with reduced hallucinations.

## Overview

This project generates product review summaries at two CEFR levels: A1 (simple paragraphs) and C1 (analytical bullet points with sentiment markers). Training combines Supervised Fine-Tuning (SFT) for format control and Direct Preference Optimization (DPO) for hallucination reduction using Qwen2.5-3B-Instruct as the base model.

## Key Features

- Automated data generation from Amazon 2023 review datasets across 10 product categories
- Multi-level summarization: A1 (beginner), C1 (advanced), and hallucinated variants
- Two-stage training: SFT for format adherence + DPO for factual accuracy
- Memory-efficient training with LoRA and 4-bit quantization
- LLM-as-judge evaluation framework

## Repository Structure

```
review_gen.ipynb     # Data generation pipeline
train_qwen.ipynb     # Training (SFT + DPO) and evaluation
```

## Requirements

```bash
pip install unsloth transformers datasets peft bitsandbytes trl
pip install emoji langdetect openai
```

## Data Generation

The `review_gen.ipynb` notebook generates training triplets from Amazon 2023 review datasets across 10 product categories:
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

For each product, it creates three summaries:

**A1 (Simple):** 3-4 short sentences in paragraph format using basic vocabulary and present tense. No bullet points.

**C1 (Complex):** 3-6 analytical bullet points with sentiment markers:
- `(+)` Consensus strengths
- `(-)` Consensus weaknesses
- `(~)` Mixed/controversial opinions

Each point is 18-30 words with sophisticated vocabulary.

**C1 Hallucinated:** Same structure as C1 but contains exactly ONE subtle factual error:
- Attribute error: Non-existent feature
- Quantity error: Wrong measurements
- Sentiment error: Reversed consensus
- Comparison error: Unsupported claim

The pipeline downloads and decompresses Amazon review data, cleans and samples 5-30 reviews per product, generates summaries via LLM API (Qwen-Plus), validates format and quality, then saves JSONL files per category. Target is 200 products per category (2,000 total).

### Output Format

```json
{
  "parent_asin": "B089Q5MJ2K",
  "reviews": [...],
  "a1_summary": "Most people like this laptop...",
  "c1_summary": "(+) Exceptional performance...\n(-) Battery concerns...",
  "c1_hallucinated": "(+) Exceptional performance...\n(-) Poor screen quality..."
}
```

## Training

The `train_qwen.ipynb` notebook implements two-stage training on Qwen2.5-3B-Instruct with interactive testing and automated evaluation.

### Stage 1 - Supervised Fine-Tuning (SFT)

Teaches the model format adherence and readability control for both A1 and C1 styles.

**Configuration:**
- Base model: Qwen2.5-3B-Instruct
- LoRA rank: 16, alpha: 16
- Quantization: 4-bit (QLoRA)
- Learning rate: 5e-5
- Epochs: 2
- Max sequence length: 6144 tokens

### Stage 2 - Direct Preference Optimization (DPO)

Applies preference learning where factual C1 summaries are preferred over hallucinated ones.

**Configuration:**
- Learning rate: 5e-6
- Beta: 0.1
- Epochs: 1
- Same LoRA and quantization settings


## Evaluation

Automated evaluation uses Qwen-Plus as an LLM judge on 50 held-out products (5 per category).

**Scoring criteria:**
- Score: 0-10
- Factuality: Evidence-based claims vs fabrication/sentiment reversal
- Format compliance: Paragraph (A1) vs bullet points with markers (C1)

**Output:** JSON file with scores, factuality flags, and detected hallucinations for each sample.


## Data Sources

- Reviews: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
