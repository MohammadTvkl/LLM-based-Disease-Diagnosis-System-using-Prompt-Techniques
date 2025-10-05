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

---

## 📂 Project Files Description

| File / Folder | Description |
|----------------|-------------|
| **medical-ai-chatbot-front.zip** | Frontend code (Next.js). In `page.tsx`, set the variable `API_URL` to your NGROK URL. |
| **Medical_Assistant-final.ipynb** | Main backend notebook (server side). Loads Meerkat-7B (float16). Uses `NGROK_AUTH_TOKEN` to connect frontend ↔ backend on Google Colab. Handles input parsing, reasoning extraction, and response formatting. |
| **Meerkat_Model.ipynb** | Loads Meerkat-7B (float16), cleans DxBench dataset, and tests on full dataset using 3 prompting techniques (Zero-Shot, CoT, Least-to-Most). Saves model outputs and creates a *judge* file for GPT-5 evaluation. |
| **bnb_4bit_smash_Meerkat_Model.ipynb** | Loads quantized (4-bit) Meerkat-7B, runs tests on selected samples, saves outputs and *judge* files for GPT-5. Optimized for low-resource inference. |
| **GPT_4o_test_creating_prompts.ipynb** | Generates prompt formats for testing GPT-4o as a comparison model. Outputs are later used by API scripts for evaluation. |
| **test-api-final.py** | Script for sending model outputs to **Metis API (GPT-5 Judge)**. Requires `api_key` and `bot_id`. Takes `verify` files from previous notebooks, sends them to GPT-5, and saves JSONL responses. |
| **generate.py** | Runs GPT-4o (via API) using prompts generated earlier and stores outputs, which are later evaluated by GPT-5 Judge via `test-api-final.py`. |
| **dep-analyze.py** | Analyzes the GPT-5 JSONL results to compute Top-1 / Top-3 / Top-5 accuracy and per-department performance. Adjust input/output paths before running. |

---

## 🧠 Notes
- Run notebooks on **Google Colab**.
- Always execute `!pip install -U datasets` and then **Restart Runtime** before running cells.
- Keep NGROK active during frontend-backend connection testing.
