# LLM-based-Disease-Diagnosis-System-using-Prompt-Techniques

## Overview

This project aims to enhance the performance of **DiagnosisGPT-6B** (`FreedomIntelligence/DiagnosisGPT-6B`) by testing various **Prompt Engineering** techniques. The goal is to identify an optimal prompt structure that improves disease prediction accuracy.

## Project Objectives

- Experiment with different **prompt engineering techniques** (e.g., **Chain-of-Thought (CoT), Zero-shot, Few-shot**).
- Introduce a **base prompt** that, when appended to user queries, improves diagnostic accuracy.
- Evaluate the effectiveness of different prompt strategies with accuracy comparisons.
- Develop an **interactive interface** where users can input symptoms and receive potential diagnoses.

## Model Used

- **Model Name**: [DiagnosisGPT-6B](https://huggingface.co/FreedomIntelligence/DiagnosisGPT-6B)
- **Framework**: Hugging Face `transformers`
- **Architecture**: Causal Language Model (LLM) fine-tuned for medical diagnosis.

## Dataset & Testing Approach

- Various prompt strategies are tested using real-world disease symptom queries.
- Performance is measured based on model accuracy, response coherence, and medical relevance.

## Installation & Setup

To run this project locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/LLM-based-Disease-Diagnosis-System.git
   cd LLM-based-Disease-Diagnosis-System
   ```

2. Install dependencies:

   ```bash
   pip install torch transformers
   ```

3. Load the model and tokenizer:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "FreedomIntelligence/DiagnosisGPT-6B"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```

## Running the Model

Run a basic query for disease diagnosis:

```python
question = "What are the common symptoms of diabetes?"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Q: {question}\nA: {answer}")
```

## Prompt Engineering Strategies To be Tested

1. **Zero-Shot Prompting**
2. **Few-Shot Prompting**
3. **Chain-of-Thought (CoT) Prompting**
4. **Instruction-Tuned Prompting**
5. **Custom Base Prompting** (Introduced for this project)

## Evaluation Metrics

- **Accuracy**: Percentage of correct diagnoses based on ground truth.
- **Response Coherence**: Whether the model provides structured, meaningful outputs.
- **Medical Validity**: Expert validation of responses (if applicable).

## Future Work

- Further optimization of prompt techniques.
- Integrating user feedback mechanisms.
