# 🔬 GRPO Fine-Tuning of Qwen3-1.7B on PHYBench Physics Benchmark with CoT Reasoning

This project showcases a **fine-tuning pipeline** for the [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) language model using **Group Relative Policy Optimization (GRPO)** on a domain-specific benchmark dataset called [PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench). The goal is to enhance **step-by-step physics problem solving** using **Chain-of-Thought (CoT)** reasoning.

---

## 🚀 Objective

Physics questions from standardized exams often require **multi-step symbolic reasoning** rather than shallow factual retrieval. This project focuses on:

- Teaching the model to **reason step-by-step** through **CoT prompting**
- Rewarding the model not only for **final correctness**, but also for:
  - ✅ **Boxed answer formatting**
  - ✅ **Sufficient step-wise trace**
  - ✅ **Symbolic expression similarity**

The final model is exported in **full FP32 precision** for deployment or further research.

---

## 🧠 Methodology

### 1. Dataset: [PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench)

- Contains expert-annotated physics problems with detailed solution traces.
- Adapted for CoT-style prompting using a custom chat template.

### 2. Base Model

- [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) – A strong open-source multilingual LLM.
- Quantized using **4-bit NF4** for efficient fine-tuning with LoRA adapters.

### 3. Training Framework

- Utilizes [Unsloth](https://github.com/unslothai/unsloth) + [Transformers Reinforcement Learning (TRL)](https://github.com/huggingface/trl)
- Training algorithm: **Group Relative Policy Optimization (GRPO)**
- LoRA for parameter-efficient fine-tuning.

### 4. Reward Function

The training loop uses a **multi-component reward function**:

```python
reward = 
  w_format  × (presence of \boxed{}) +
  w_trace   × (minimum number of line breaks for reasoning) +
  w_correct × (symbolic equivalence via SymPy)
