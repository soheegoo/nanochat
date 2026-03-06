"""
Needle in a Haystack Evaluation
================================
Compares two model checkpoints on long-context retrieval.

Tests whether the model can retrieve a specific fact ("needle") hidden
inside a long passage of filler text ("haystack") at three positions:
beginning, middle, and end.
"""

import os
import random
import argparse
import json

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model, load_model_from_dir
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Needle templates
# ---------------------------------------------------------------------------

NEEDLES = [
    ("The secret code for this document is {code}.", "The secret code for this document is"),
    ("The magic word hidden in this text is {code}.", "The magic word hidden in this text is"),
    ("Remember this number: {code}. It is the key.", "Remember this number:"),
    ("The password you are looking for is {code}.", "The password you are looking for is"),
]

CODES = [
    "BANANA97", "ZEBRA42", "XYLOPHONE11", "JUPITER88",
    "CRIMSON55", "THUNDER23", "AVALON77", "PHANTOM09",
]

# ---------------------------------------------------------------------------
# Haystack builder
# ---------------------------------------------------------------------------

def build_haystack(tokenizer, total_tokens: int, needle_tokens: list, position: str) -> list:
    """
    Build a token sequence of approximately total_tokens length with the
    needle inserted at the specified position (beginning, middle, end).
    """
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Scientists have discovered that regular exercise improves cognitive function. "
        "The history of mathematics spans thousands of years across many cultures. "
        "In the year 1969, humans first landed on the moon. "
        "Climate change is one of the most pressing issues of our time. "
    )
    filler_tokens = tokenizer.encode(filler)

    n_needle = len(needle_tokens)
    n_filler_needed = max(total_tokens - n_needle, 0)

    haystack_tokens = []
    while len(haystack_tokens) < n_filler_needed:
        haystack_tokens.extend(filler_tokens)
    haystack_tokens = haystack_tokens[:n_filler_needed]

    if position == "beginning":
        return needle_tokens + haystack_tokens
    elif position == "end":
        return haystack_tokens + needle_tokens
    elif position == "middle":
        mid = len(haystack_tokens) // 2
        return haystack_tokens[:mid] + needle_tokens + haystack_tokens[mid:]
    else:
        raise ValueError(f"Unknown position: {position}")

# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_needle(model, tokenizer, total_tokens: int, position: str, seed: int = 42) -> dict:
    """
    Run a single needle-in-a-haystack trial.
    Returns a dict. If the context exceeds the model's sequence length,
    the example is skipped (skipped=True) rather than truncated.
    """
    rng = random.Random(seed)
    needle_template, query_stem = rng.choice(NEEDLES)
    code = rng.choice(CODES)

    needle_text = needle_template.format(code=code)
    needle_tokens = tokenizer.encode(" " + needle_text)

    context_tokens = build_haystack(tokenizer, total_tokens, needle_tokens, position)
    query_tokens = tokenizer.encode(" " + query_stem)
    input_tokens = context_tokens + query_tokens

    max_len = model.config.sequence_len
    if len(input_tokens) > max_len:
        return {
            "position": position,
            "total_tokens": total_tokens,
            "correct": False,
            "skipped": True,
            "predicted": None,
            "expected": code,
            "needle": needle_text,
            "query": query_stem,
            "context_length_used": len(input_tokens),
        }

    device = model.get_device()
    ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        logits = model(ids)
    next_token_logits = logits[0, -1, :]
    predicted_token = next_token_logits.argmax().item()
    predicted_text = tokenizer.decode([predicted_token]).strip()

    code_tokens = tokenizer.encode(" " + code)
    expected_token = code_tokens[0]
    correct = (predicted_token == expected_token)

    return {
        "position": position,
        "total_tokens": total_tokens,
        "correct": correct,
        "skipped": False,
        "predicted": predicted_text,
        "expected": code,
        "needle": needle_text,
        "query": query_stem,
        "context_length_used": len(input_tokens),
    }

# ---------------------------------------------------------------------------
# Full evaluation sweep
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, num_trials: int = 10) -> dict:
    """
    Run the full needle eval across positions and context lengths.
    Returns a nested dict: results[position][context_length] = accuracy or None (skipped)
    """
    positions = ["beginning", "middle", "end"]
    context_lengths = [256, 512, 768, 1024, 1536, 2000]

    results = {}
    for position in positions:
        results[position] = {}
        for ctx_len in context_lengths:
            correct = 0
            skipped = 0
            for trial in range(num_trials):
                result = evaluate_needle(model, tokenizer, ctx_len, position, seed=trial * 100 + hash(position) % 100)
                if result["skipped"]:
                    skipped += 1
                elif result["correct"]:
                    correct += 1

            if skipped == num_trials:
                results[position][ctx_len] = None
                print(f"  position={position:10s} ctx_len={ctx_len:5d} accuracy=N/A (exceeds model seq len)")
            else:
                evaluated = num_trials - skipped
                accuracy = correct / evaluated
                results[position][ctx_len] = accuracy
                print(f"  position={position:10s} ctx_len={ctx_len:5d} accuracy={accuracy:.2f} ({correct}/{evaluated})")

    return results

def compute_summary(results: dict) -> dict:
    """Compute per-position and overall accuracy summaries, ignoring skipped (None) entries."""
    summary = {}
    all_accs = []
    for position, ctx_results in results.items():
        accs = [v for v in ctx_results.values() if v is not None]
        avg = sum(accs) / len(accs) if accs else float('nan')
        summary[position] = avg
        all_accs.extend(accs)
    summary["overall"] = sum(all_accs) / len(all_accs) if all_accs else float('nan')
    return summary

# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_comparison_table(results1: dict, results2: dict, tag1: str, tag2: str):
    positions = ["beginning", "middle", "end"]
    context_lengths = sorted(next(iter(results1.values())).keys())

    print("\n" + "=" * 80)
    print(f"NEEDLE IN A HAYSTACK — COMPARISON")
    print(f"  Model 1: {tag1} (seq_len=512)")
    print(f"  Model 2: {tag2} (seq_len=2048)")
    print("=" * 80)

    for position in positions:
        print(f"\nPosition: {position.upper()}")
        print(f"  {'ctx_len':>8}  {'Model1':>10}  {'Model2':>10}  {'Delta':>10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
        for ctx_len in context_lengths:
            a1 = results1[position].get(ctx_len)
            a2 = results2[position].get(ctx_len)
            a1_str = f"{a1:.2f}" if a1 is not None else "N/A"
            a2_str = f"{a2:.2f}" if a2 is not None else "N/A"
            if a1 is not None and a2 is not None:
                delta_str = f"{a2 - a1:+.2f}"
            else:
                delta_str = "N/A"
            print(f"  {ctx_len:>8}  {a1_str:>10}  {a2_str:>10}  {delta_str:>10}")

    s1 = compute_summary(results1)
    s2 = compute_summary(results2)
    print(f"\nOVERALL SUMMARY (N/A entries excluded from averages)")
    print(f"  {'position':>12}  {'Model1':>10}  {'Model2':>10}  {'Delta':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in positions + ["overall"]:
        v1, v2 = s1[k], s2[k]
        v1_str = f"{v1:.2f}" if v1 == v1 else "N/A"
        v2_str = f"{v2:.2f}" if v2 == v2 else "N/A"
        delta_str = f"{v2 - v1:+.2f}" if v1 == v1 and v2 == v2 else "N/A"
        print(f"  {k:>12}  {v1_str:>10}  {v2_str:>10}  {delta_str:>10}")
    print("=" * 80 + "\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model_for_eval(model_tag, step, device):
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, "eval", model_tag=model_tag, step=step)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Needle in a haystack eval")
    parser.add_argument("--model-tag", type=str, default=None, help="model tag for single eval")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (None = latest)")
    parser.add_argument("--num-trials", type=int, default=10, help="trials per (position, ctx_len) cell")
    parser.add_argument("--compare", action="store_true", help="compare two models")
    parser.add_argument("--model-tag-1", type=str, default=None)
    parser.add_argument("--step-1", type=int, default=None)
    parser.add_argument("--model-tag-2", type=str, default=None)
    parser.add_argument("--step-2", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    if args.compare:
        assert args.model_tag_1 and args.model_tag_2, "Need --model-tag-1 and --model-tag-2 for comparison"

        print(f"\nLoading model 1: {args.model_tag_1} (step={args.step_1})")
        model1, tokenizer = load_model_for_eval(args.model_tag_1, args.step_1, device)
        print(f"Running needle eval for model 1...")
        results1 = run_eval(model1, tokenizer, args.num_trials)
        del model1
        torch.cuda.empty_cache()

        print(f"\nLoading model 2: {args.model_tag_2} (step={args.step_2})")
        model2, tokenizer = load_model_for_eval(args.model_tag_2, args.step_2, device)
        print(f"Running needle eval for model 2...")
        results2 = run_eval(model2, tokenizer, args.num_trials)
        del model2

        print_comparison_table(results1, results2, args.model_tag_1, args.model_tag_2)

        base_dir = get_base_dir()
        out_path = os.path.join(base_dir, "needle_eval_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "model1": args.model_tag_1, "step1": args.step_1,
                "model2": args.model_tag_2, "step2": args.step_2,
                "results1": {p: {str(k): v for k, v in r.items()} for p, r in results1.items()},
                "results2": {p: {str(k): v for k, v in r.items()} for p, r in results2.items()},
            }, f, indent=2)
        print(f"Results saved to {out_path}")

    else:
        assert args.model_tag, "Need --model-tag for single eval"
        print(f"\nLoading model: {args.model_tag} (step={args.step})")
        model, tokenizer = load_model_for_eval(args.model_tag, args.step, device)
        print(f"Running needle eval...")
        results = run_eval(model, tokenizer, args.num_trials)
        summary = compute_summary(results)
        print(f"\nSummary: {summary}")

if __name__ == "__main__":
    main()