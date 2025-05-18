#!/usr/bin/env python
"""
eval_phybench.py – quick-and-clean evaluator for GRPO-finetuned Qwen models
==========================================================================

Reports the four headline metrics used in the README table:

• Mean Total Reward (w_format · F + w_trace · T + w_correct · C)
• Format accuracy      – share of completions that contain \boxed{…}
• Trace-length accuracy – share with ≥ 6 newline-separated reasoning steps
• EED Correct           – SymPy expression-equivalence ≥ 0.9

The weights (w_format = 0.2, w_trace = 0.3, w_correct = 0.5) match
the training script.  Tweaking them for ablations?  Just change below.

Requirements
------------
pip install transformers datasets sympy tqdm accelerate --upgrade
"""
from __future__ import annotations
import argparse, re, math, torch, sympy as sp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── reward weights – keep in sync with training ──────────────────────────
W_FORMAT  = 0.2        # presence of \boxed{…}
W_TRACE   = 0.3        # ≥ 6 reasoning lines
W_CORRECT = 0.5        # expression-equivalence distance (EED)

BOX_RE = re.compile(r"\\boxed\{([^}]+)\}")

# ── helpers ───────────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    """Return the *last* boxed expression in a string; blank if none found."""
    m = BOX_RE.findall(text or "")
    return m[-1].strip() if m else ""

def eed_score(gen: str, ref: str) -> float:
    """Expression-Equivalence Distance in [0, 1]. 1 = symbolically identical."""
    try:
        if sp.simplify(gen) == sp.simplify(ref):
            return 1.0
        g, r = sp.sympify(gen), sp.sympify(ref)
        size = lambda e: len(e.free_symbols) + len(list(e.preorder_traversal()))
        rel  = abs(size(g) - size(r)) / (size(r) + 1e-6)
        return max(0.0, (0.6 - rel) / 0.6)          # same tolerance as train
    except Exception:
        return 0.0

def mean(xs): return sum(xs) / len(xs) if xs else 0.0

# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  required=True,  help="HF path or local dir")
    ap.add_argument("--dataset_name",required=True,  help="HF dataset repo")
    ap.add_argument("--split_full",  required=True,  help="JSON file inside repo")
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--device", default="auto",
                    help='"cpu", "cuda", "-1" or accelerate "auto"')
    args = ap.parse_args()

    # 1️⃣  model + tokenizer ------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    # 2️⃣  load PHYBench subset -------------------------------------------
    ds = load_dataset(args.dataset_name,
                      data_files=args.split_full,
                      split="train")

    # 3️⃣  iterate ---------------------------------------------------------
    fmt_hits = trace_hits = eed_hits = 0
    rewards  = []
    for idx in tqdm(range(0, len(ds), args.batch_size), ncols=88):
        batch = ds[idx : idx + args.batch_size]
        prompts = batch["prompt"]
        answers = [extract_answer(a) for a in batch["answer"]]

        toks = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outs = model.generate(**toks,
                                  max_new_tokens=args.max_new_tokens,
                                  do_sample=False, temperature=0.0)
        for p, out, ref in zip(prompts, outs, answers):
            comp = tok.decode(out, skip_special_tokens=True)
            gen  = comp[len(p):]                  # completion only

            has_box   = "\\boxed{" in gen
            long_trace= gen.count("\n") >= 6
            eed       = eed_score(extract_answer(gen), ref)

            fmt_hits   += has_box
            trace_hits += long_trace
            eed_hits   += (eed >= 0.9)

            reward = W_FORMAT*has_box + W_TRACE*long_trace + W_CORRECT*eed
            rewards.append(reward)

    n = len(ds)
    print("\n──────── Evaluation on", args.split_full, "────────")
    print(f"Mean Total Reward : {mean(rewards):.3f}")
    print(f"Format accuracy   : {fmt_hits/n:.2%}")
    print(f"Trace ≥ 6 accuracy: {trace_hits/n:.2%}")
    print(f"EED Correct (≥0.9): {eed_hits/n:.2%}")
    print("────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
