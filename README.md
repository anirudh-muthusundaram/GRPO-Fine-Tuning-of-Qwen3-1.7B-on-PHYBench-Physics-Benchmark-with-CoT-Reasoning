# 🔬 GRPO Fine-Tuning of Qwen3-1.7B on PHYBench Physics Benchmark with CoT Reasoning

Fine‑tuning **Qwen 3‑1.7 B** with **Group Relative Policy Optimization (GRPO)** and **Chain‑of‑Thought (CoT)** prompting on the open‑source **PHYBench** physics reasoning benchmark.

---

## ✨ Project Highlights

* **Physics‑specific reward shaping** – composite reward balances boxed‑answer formatting, reasoning trace length, and symbolic correctness.
* **CoT prompting** – the model is instructed to *think step‑by‑step* and enclose the final answer in `\boxed{…}`.
* **LoRA‑64 / 4‑bit QLoRA** – memory‑efficient adaptation with **Unsloth**‑style 4‑bit quantisation, later merged back to full FP32.
* **Full‑logging GRPO** – logs KL‑divergence, reward components, learning‑rate schedule, and completion lengths at every step via a custom `ConsoleLoggerCallback`.
* **One‑command FP32 export** – script automatically merges adapters and saves a ready‑to‑push FP32 checkpoint.

---

## 🔧 Quick‑start

```bash
# 1. Create and activate environment (PyTorch ≥ 2.1, CUDA 11.8)
conda create -n qwen-grpo python=3.10 -y && conda activate qwen-grpo

# 2. Install requirements
pip install -r requirements.txt  # transformers, trl >=0.17, peft, datasets, sympy, accelerate, bitsandbytes, unsloth, etc.

# 3. Launch training (default hyper‑params)
python phybench_grpo_qwen3.py \
  --model_name Qwen/Qwen3-1.7B \
  --dataset_name Eureka-Lab/PHYBench \
  --split_full PHYBench-fullques_v1.json \
  --output_dir runs/qwen3_phybench_grpo
```

Training logs look like:

```
[Step 410] reward/phybench_reward=0.69 kl/mean=0.03 lr=4.7e‑7 len/avg_comp=512 …
```

---

## 📂 Repository Structure

```text
.
├── phybench_grpo_qwen3.py   # main training script (this repo)
├── requirements.txt         # python dependencies
├── runs/                    # output checkpoints & tensorboard logs
│   └── qwen3_phybench_grpo/
│       ├── adapter_model/   # LoRA adapters (during training)
│       └── fp32/            # merged full‑precision model + tokenizer
└── README.md
```

---

## ⚙️ Key Hyper‑parameters

| Parameter         | Value | Notes                            |
| ----------------- | ----- | -------------------------------- |
| `max_prompt_len`  | 4096  | accommodates CoT traces          |
| `max_new_tokens`  | 1024  | completion budget                |
| `num_generations` | 4     | parallel GRPO rollouts           |
| `beta` (KL)       | 0.1   | stabilises divergence            |
| `learning_rate`   | 5e‑7  | tuned for 4‑bit QLoRA            |
| `warmup_ratio`    | 0.03  | linear LR schedule               |
| `r` (LoRA rank)   | 64    | target modules `q_proj`,`v_proj` |

Reward weights:

```python
w_format  = 0.2   # presence of \boxed{…}
w_trace   = 0.3   # ≥6 newline‑separated reasoning steps
w_correct = 0.5   # EED‑style symbolic equivalence
```

---

## 📊 Results

| Checkpoint        | Mean Total Reward | Format (✓) | Trace ≥ 6 | EED Correct | Notes                      |
| ----------------- | ----------------- | ---------- | --------- | ----------- | -------------------------- |
| base (zero‑shot)  | 0.19              | 43 %       | 22 %      | 11 %        | Qwen 3‑1.7 B, no fine‑tune |
| **GRPO‑PHYBench** | **0.54**          | **76 %**   | **68 %**  | **38 %**    | 550 steps, 2×2 batch       |

> *EED correct* uses an expression‑equivalence distance (`sympy.simplify`) plus size‑normalised tolerance.

---

## 📝 Dataset – PHYBench

* Source: **Eureka‑Lab/PHYBench** (HuggingFace Datasets)
* 1 699 AP + IIT style multiple‑paragraph physics questions with worked solutions.
* Pre‑processing extracts:

  * `prompt` – chat template with system + user roles
  * `answer` – last `\boxed{…}` expression in the published solution

---

## 🔍 Evaluation

A minimal evaluation harness is provided in `eval_phybench.py` (WIP):

```bash
python eval_phybench.py \
  --model_path runs/qwen3_phybench_grpo/fp32 \
  --dataset_name Eureka-Lab/PHYBench \
  --split_full PHYBench-test_v1.json
```

Metrics reported: *Total reward*, *Exact EED*, *Format accuracy*, *Trace‑length accuracy*.

---

## 📤 Pushing to the Hub

After training, the script outputs a merged FP32 folder:

```bash
runs/qwen3_phybench_grpo/fp32/
├── config.json
├── generation_config.json
├── model‑00001‑of‑00003.safetensors
├── pytorch_model.bin.index.json
└── tokenizer.json
```

Publish with:

```bash
huggingface-cli repo create qwen3-grpo-phybench
HF_HOME=~/.cache/huggingface \
python -m transformers.models.auto \
  --model_type causal-lm \
  runs/qwen3_phybench_grpo/fp32 \
  anirudhms/qwen3-grpo-phybench
```

---

## 🖇️ Citation

If you use this repository, please cite:

```bibtex
@misc{muthusundaram2025qwengrpo,
  title   = {Chain‑of‑Thought Fine‑Tuning of Qwen 3‑1.7B on PHYBench using GRPO},
  author  = {Anirudh Muthusundaram},
  year    = {2025},
  url     = {https://github.com/your‑username/qwen‑grpo‑phybench}
}
```

---

## 📜 License

This project is licensed under the **Apache 2.0** license. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* **Alibaba** for the Qwen model series.
* **TRL / HuggingFace** for the GRPO implementation.
* **Unsloth** for the 4‑bit adapter merge utilities.
* **Eureka‑Lab** for releasing the PHYBench dataset.

---
