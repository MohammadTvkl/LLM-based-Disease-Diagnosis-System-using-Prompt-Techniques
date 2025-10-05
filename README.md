# 🧠 LLM-based Disease Diagnosis System using Prompt Engineering

This project designs a **lightweight medical assistant** using **Meerkat-7B**, a 7B-parameter open medical LLM,
to provide efficient and explainable disease diagnosis **without internet access** (privacy-preserving).

---

## 🚀 Features
- 🩺 Disease diagnosis based on patient symptoms
- 🧩 Prompt engineering techniques:
  - Zero-Shot Prompting  
  - Single-Step Chain-of-Thought  
  - Least-to-Most Prompting
- 📊 Evaluation metrics: Top-1 / Top-3 / Top-5 accuracy
- ⚙️ Lightweight model: Runs on local hardware (Quantized Meerkat-7B)
- 🔒 Privacy preserved – no external API calls

---

## 🧬 Technologies
- Python, PyTorch, Hugging Face Transformers  
- Prompt Engineering, LLM Reasoning  
- Dataset: [DxBench](https://huggingface.co/datasets/FreedomIntelligence/DxBench)

---

## 📊 Results Summary
| Model | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| GPT-4o | 32% | 50% | 63% |
| Meerkat-7B (FP16) | 33% | 53% | 56% |
| Meerkat-7B (4-bit Quantized) | 29% | 52% | 55% |

---
