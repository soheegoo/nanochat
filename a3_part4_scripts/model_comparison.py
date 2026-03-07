"""
Compare two models side-by-side on benchmark questions.

This script runs both models on the same questions and shows you:
- Which questions nanochat answered correctly that picochat missed
- Per-question comparison of outputs
- Overall win/loss statistics

Usage:
    python -m scripts.model_comparison \
        --model1-tag picochat-combo-mar-1-maryam \
        --model2-tag nanochat-d20-kashish \
        --task-name MMLU \
        --source sft
"""

import argparse
import json
from functools import partial
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


def compare_models_on_task(
    task_object,
    model1, tokenizer1, engine1, model1_name,
    model2, tokenizer2, engine2, model2_name,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    batch_size=8,
    max_problems=None,
):
    """
    Run both models on the same task and compare results.
    
    Returns:
        dict with comparison stats and per-problem results
    """
    device = model1.get_device()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    
    results = []
    model1_correct = 0
    model2_correct = 0
    model2_wins = 0  # model2 correct, model1 wrong
    
    print0(f"\nEvaluating {num_problems} problems from {task_object.__class__.__name__}...")
    print0("-" * 70)
    
    for i in range(num_problems):
        conversation = task_object[i]
        
        # Extract question text from conversation messages
        if 'messages' in conversation and len(conversation['messages']) > 0:
            question = conversation['messages'][0]['content']
        else:
            question = conversation.get('prompt', conversation.get('problem', 'N/A'))
        
        # Evaluate model 1
        if task_object.eval_type == 'generative':
            encoded_prompt = tokenizer1.render_for_completion(conversation)
            completions1, _ = engine1.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            prefix_length = len(encoded_prompt)
            completions1_text = [tokenizer1.decode(c[prefix_length:]) for c in completions1]
            outcomes1 = [task_object.evaluate(conversation, c) for c in completions1_text]
            passed1 = any(outcomes1)
            answer1 = completions1_text[0]
        else:  # categorical
            encoded_prompt = tokenizer1.render_for_completion(conversation)
            with torch.no_grad():
                logits1 = model1(torch.tensor([encoded_prompt], device=device))
            
            letters = conversation['letters']
            letter_ids = [tokenizer1.encode(letter)[0] for letter in letters]
            answer_pos = len(encoded_prompt) - 1
            focus_logits = logits1[0, answer_pos, letter_ids]
            predicted_idx = focus_logits.argmax().item()
            answer1 = letters[predicted_idx]
            passed1 = task_object.evaluate(conversation, answer1)
        
        # Evaluate model 2
        if task_object.eval_type == 'generative':
            encoded_prompt = tokenizer2.render_for_completion(conversation)
            completions2, _ = engine2.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            prefix_length = len(encoded_prompt)
            completions2_text = [tokenizer2.decode(c[prefix_length:]) for c in completions2]
            outcomes2 = [task_object.evaluate(conversation, c) for c in completions2_text]
            passed2 = any(outcomes2)
            answer2 = completions2_text[0]
        else:  # categorical
            encoded_prompt = tokenizer2.render_for_completion(conversation)
            with torch.no_grad():
                logits2 = model2(torch.tensor([encoded_prompt], device=device))
            
            letters = conversation['letters']
            letter_ids = [tokenizer2.encode(letter)[0] for letter in letters]
            answer_pos = len(encoded_prompt) - 1
            focus_logits = logits2[0, answer_pos, letter_ids]
            predicted_idx = focus_logits.argmax().item()
            answer2 = letters[predicted_idx]
            passed2 = task_object.evaluate(conversation, answer2)
        
        # Track results
        model1_correct += int(passed1)
        model2_correct += int(passed2)
        
        if passed2 and not passed1:
            model2_wins += 1
            status = f"✓ {model2_name} WIN"
        elif passed1 and not passed2:
            status = f"✗ {model1_name} WIN"
        elif passed1 and passed2:
            status = "✓ BOTH"
        else:
            status = "✗ BOTH FAIL"
        
        results.append({
            'problem_id': i,
            'question': question,  # Full question text
            'model1_answer': str(answer1),  # Full answer
            'model1_correct': passed1,
            'model2_answer': str(answer2),  # Full answer
            'model2_correct': passed2,
            'status': status,
        })
        
        # Progress
        print(f"\r\033[KProgress: {i+1}/{num_problems} | {model2_name} wins: {model2_wins}", end='', flush=True)
    
    print()  # Newline after progress
    
    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_accuracy': model1_correct / num_problems,
        'model2_accuracy': model2_correct / num_problems,
        'model2_wins': model2_wins,
        'total_problems': num_problems,
        'per_problem_results': results,
    }


def print_comparison_summary(comparison_results):
    """Print a nice summary of the comparison."""
    model1_name = comparison_results['model1_name']
    model2_name = comparison_results['model2_name']
    model1_acc = comparison_results['model1_accuracy']
    model2_acc = comparison_results['model2_accuracy']
    model2_wins = comparison_results['model2_wins']
    total = comparison_results['total_problems']
    
    print0("\n" + "=" * 70)
    print0("COMPARISON SUMMARY")
    print0("=" * 70)
    print0(f"\n{model1_name}:")
    print0(f"  Accuracy: {model1_acc*100:.2f}% ({int(model1_acc*total)}/{total})")
    
    print0(f"\n{model2_name}:")
    print0(f"  Accuracy: {model2_acc*100:.2f}% ({int(model2_acc*total)}/{total})")
    
    print0(f"\n{model2_name} unique wins: {model2_wins}")
    print0(f"  (Questions {model2_name} got right that {model1_name} missed)")
    
    improvement = ((model2_acc - model1_acc) / model1_acc) * 100 if model1_acc > 0 else float('inf')
    print0(f"\nRelative improvement: {improvement:+.1f}%")
    print0("=" * 70)


def print_model2_wins(comparison_results, max_show=10, max_answer_length=None):
    """Print questions where model2 won but model1 failed."""
    model1_name = comparison_results['model1_name']
    model2_name = comparison_results['model2_name']
    wins = [r for r in comparison_results['per_problem_results'] if r['model2_correct'] and not r['model1_correct']]
    
    if not wins:
        print0(f"\n{model2_name} did not have any unique wins.")
        return
    
    print0(f"\n{'='*80}")
    print0(f"QUESTIONS WHERE {model2_name.upper()} WINS")
    print0(f"({'Showing first ' + str(max_show) if len(wins) > max_show else f'All {len(wins)} wins'})")
    print0("=" * 80)
    
    for idx, win in enumerate(wins[:max_show]):
        print0(f"\n{'─'*80}")
        print0(f"[Win {idx+1}/{min(len(wins), max_show)}] Problem #{win['problem_id']}")
        print0(f"{'─'*80}")
        
        # Format question nicely (wrap long lines)
        question_lines = win['question'].split('\n')
        for line in question_lines:
            if len(line) <= 76:
                print0(f"  {line}")
            else:
                # Wrap long lines at word boundaries
                words = line.split()
                current_line = "  "
                for word in words:
                    if len(current_line) + len(word) + 1 <= 76:
                        current_line += word + " "
                    else:
                        print0(current_line.rstrip())
                        current_line = "  " + word + " "
                if current_line.strip():
                    print0(current_line.rstrip())
        
        # Format answers (handle multi-line answers)
        model1_answer = win['model1_answer']
        model2_answer = win['model2_answer']
        
        # Truncate if requested
        if max_answer_length:
            if len(model1_answer) > max_answer_length:
                model1_answer = model1_answer[:max_answer_length] + "..."
            if len(model2_answer) > max_answer_length:
                model2_answer = model2_answer[:max_answer_length] + "..."
        
        print0(f"\n  {model1_name} answer: ✗")
        for line in model1_answer.split('\n'):
            print0(f"    {line}")
        
        print0(f"\n  {model2_name} answer: ✓")
        for line in model2_answer.split('\n'):
            print0(f"    {line}")
    
    print0("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two models')
    parser.add_argument('--model1-tag', type=str, required=True, help='First model tag (baseline)')
    parser.add_argument('--model2-tag', type=str, required=True, help='Second model tag (to compare)')
    parser.add_argument('--model1-step', type=int, default=None, help='Step for model1')
    parser.add_argument('--model2-step', type=int, default=None, help='Step for model2')
    parser.add_argument('-i', '--source', type=str, default='sft', help='Model source: base|sft|rl')
    parser.add_argument('-a', '--task-name', type=str, required=True, help='Task name: MMLU|ARC-Easy|ARC-Challenge|GSM8K|HumanEval|SpellingBee')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-x', '--max-problems', type=int, default=None)
    parser.add_argument('--show-wins', type=int, default=10, help='Number of model2 wins to show')
    parser.add_argument('--max-answer-length', type=int, default=None, help='Truncate answers to N chars (None = show full answers)')
    parser.add_argument('--save-json', type=str, default=None, help='Save detailed results to JSON file')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'])
    args = parser.parse_args()
    
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Load both models
    print0(f"Loading {args.model1_tag}...")
    model1, tokenizer1, meta1 = load_model(args.source, device, phase="eval", model_tag=args.model1_tag, step=args.model1_step)
    engine1 = Engine(model1, tokenizer1)
    
    print0(f"Loading {args.model2_tag}...")
    model2, tokenizer2, meta2 = load_model(args.source, device, phase="eval", model_tag=args.model2_tag, step=args.model2_step)
    engine2 = Engine(model2, tokenizer2)
    
    # Create task
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[args.task_name]
    task_object = task_module()
    
    # Run comparison
    print0(f"\n{'='*70}")
    print0(f"COMPARING: {args.model1_tag} vs {args.model2_tag}")
    print0(f"Task: {args.task_name}")
    print0(f"{'='*70}")
    
    with autocast_ctx:
        comparison_results = compare_models_on_task(
            task_object,
            model1, tokenizer1, engine1, args.model1_tag,
            model2, tokenizer2, engine2, args.model2_tag,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_problems=args.max_problems,
        )
    
    # Print summary
    print_comparison_summary(comparison_results)
    
    # Print model2 wins
    print_model2_wins(comparison_results, max_show=args.show_wins, max_answer_length=args.max_answer_length)
    
    # Save detailed results if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print0(f"\nDetailed results saved to: {args.save_json}")
