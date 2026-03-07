"""
Compare two models on custom questions loaded from a JSONL file.

This script runs both models on your custom questions and shows their outputs side-by-side.

Usage:
    python -m scripts.compare_custom \
        --model1-tag scaling_law_12_3.16e_18_try_1 \
        --model2-tag nanochat-d20-kashish \
        --questions-file custom_questions.jsonl \
        --source sft
"""

import argparse
import json
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


def load_questions(filepath):
    """Load questions from JSONL file."""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
                assert isinstance(messages, list), f"Expected list, got {type(messages)}"
                assert len(messages) >= 2, f"Need at least 2 messages"
                questions.append({"messages": messages})
            except Exception as e:
                print(f"Warning: Skipping line {line_num} due to error: {e}")
                continue
    return questions


def compare_on_custom_questions(
    questions,
    model1, tokenizer1, engine1, model1_name,
    model2, tokenizer2, engine2, model2_name,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
):
    """
    Run both models on custom questions and show side-by-side comparison.
    
    Returns:
        List of results for each question
    """
    device = model1.get_device()
    results = []
    
    print0(f"\nComparing models on {len(questions)} custom questions...")
    print0("=" * 80)
    
    for i, conversation in enumerate(questions):
        question_text = conversation['messages'][0]['content']
        
        # Generate response from model 1
        encoded_prompt = tokenizer1.render_for_completion(conversation)
        completions1, _ = engine1.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        answer1 = tokenizer1.decode(completions1[0][prefix_length:])
        
        # Generate response from model 2
        encoded_prompt = tokenizer2.render_for_completion(conversation)
        completions2, _ = engine2.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        answer2 = tokenizer2.decode(completions2[0][prefix_length:])
        
        results.append({
            'question_id': i + 1,
            'question': question_text,
            'model1_answer': answer1,
            'model2_answer': answer2,
        })
        
        # Print immediately
        print0(f"\n{'─'*80}")
        print0(f"[Question {i+1}/{len(questions)}]")
        print0(f"{'─'*80}")
        
        # Format question
        for line in question_text.split('\n'):
            print0(f"  {line}")
        
        # Show answers
        print0(f"\n  {model1_name}:")
        for line in answer1.split('\n'):
            print0(f"    {line}")
        
        print0(f"\n  {model2_name}:")
        for line in answer2.split('\n'):
            print0(f"    {line}")
        
        print0("")  # blank line for spacing
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare models on custom questions')
    parser.add_argument('--model1-tag', type=str, required=True, help='First model tag')
    parser.add_argument('--model2-tag', type=str, required=True, help='Second model tag')
    parser.add_argument('--questions-file', type=str, required=True, help='Path to JSONL file with questions')
    parser.add_argument('--model1-step', type=int, default=None, help='Step for model1')
    parser.add_argument('--model2-step', type=int, default=None, help='Step for model2')
    parser.add_argument('-i', '--source', type=str, default='sft', help='Model source: base|sft|rl')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('--save-json', type=str, default=None, help='Save results to JSON')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'])
    args = parser.parse_args()
    
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Load questions
    print0(f"Loading questions from: {args.questions_file}")
    questions = load_questions(args.questions_file)
    print0(f"Loaded {len(questions)} questions\n")
    
    # Load both models
    print0(f"Loading {args.model1_tag}...")
    model1, tokenizer1, meta1 = load_model(args.source, device, phase="eval", model_tag=args.model1_tag, step=args.model1_step)
    engine1 = Engine(model1, tokenizer1)
    
    print0(f"Loading {args.model2_tag}...")
    model2, tokenizer2, meta2 = load_model(args.source, device, phase="eval", model_tag=args.model2_tag, step=args.model2_step)
    engine2 = Engine(model2, tokenizer2)
    
    # Run comparison
    print0(f"\n{'='*80}")
    print0(f"COMPARING: {args.model1_tag} vs {args.model2_tag}")
    print0(f"Questions file: {args.questions_file}")
    print0(f"{'='*80}")
    
    with autocast_ctx:
        results = compare_on_custom_questions(
            questions,
            model1, tokenizer1, engine1, args.model1_tag,
            model2, tokenizer2, engine2, args.model2_tag,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    
    # Save if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump({
                'model1_tag': args.model1_tag,
                'model2_tag': args.model2_tag,
                'questions_file': args.questions_file,
                'results': results,
            }, f, indent=2)
        print0(f"\nResults saved to: {args.save_json}")
    
    print0("\n" + "=" * 80)
    print0(f"Comparison complete! Evaluated {len(results)} questions.")
    print0("=" * 80)
