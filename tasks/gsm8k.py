"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

# NOTE modify this for A4, or another dataset reward. Update the reward . s
    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        # BASELINE REWARD
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
    
        ####################################################################################


        # BASELINE + FORMAT
        # reward = 0.0

        # # Correctness: 1.0
        # is_correct = self.evaluate(conversation, assistant_response)
        # reward += float(is_correct)

        # # Format: 0.25 for having #### on the last line AND 2-8 newlines total
        # lines = assistant_response.rstrip().split('\n')
        # last_line = lines[-1].strip() if lines else ""
        # has_marker = last_line.startswith("####")
        # newline_count = assistant_response.count('\n')
        # good_length = 2 <= newline_count <= 8

        # if has_marker and good_length:
        #     reward += 0.25

        # return min(reward, 1.0)


        ####################################################################################

        # BASELINE + REFERENCE NUMBERS
        # reward = 0.0
        # is_correct = self.evaluate(conversation, assistant_response)
        # reward += float(is_correct)
        # # Extract all numbers from the user prompt
        # user_message = conversation['messages'][0]['content']
        # prompt_numbers = set(re.findall(r'\d+\.?\d*', user_message))
        # # Check if at least 80% of prompt numbers appear in the response
        # if prompt_numbers:
        #     matched = sum(1 for n in prompt_numbers if n in assistant_response)
        #     if matched / len(prompt_numbers) >= 0.8:
        #         reward += 0.25
        # return min(reward, 1.0)

        ####################################################################################

def _count_steps(answer_text):
    """Count the number of calculator steps (<<...>>) in a GSM8K answer."""
    return len(re.findall(r'<<[^>]+>>', answer_text))


class GSM8KFiltered(GSM8K):
    """GSM8K filtered by number of reasoning steps (calculator calls)."""

    def __init__(self, subset, split, min_steps=0, max_steps=float('inf'), **kwargs):
        # Don't pass kwargs to GSM8K yet — we need to filter first
        super().__init__(subset, split, **kwargs)
        # Filter indices by step count
        self.filtered_indices = []
        for i in range(len(self.ds)):
            num_steps = _count_steps(self.ds[i]['answer'])
            if min_steps <= num_steps <= max_steps:
                self.filtered_indices.append(i)
        print(f"GSM8KFiltered: {len(self.filtered_indices)}/{len(self.ds)} examples "
              f"with {min_steps} <= steps <= {max_steps}")

    def num_examples(self):
        return len(self.filtered_indices)

    def get_example(self, index):
        real_index = self.filtered_indices[index]
        row = self.ds[real_index]
        return self._build_conversation(row)

    def _build_conversation(self, row):
        """Shared logic to build a conversation from a dataset row."""
        question = row['question']
        answer = row['answer']
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_message_parts},
            ]
        }


class GSM8KEasy(GSM8KFiltered):
    """GSM8K problems with <= 2 reasoning steps (36% of train, 2674 problems)."""
    def __init__(self, subset, split, **kwargs):
        super().__init__(subset, split, min_steps=0, max_steps=2, **kwargs)


class GSM8KHard(GSM8KFiltered):
    """GSM8K problems with > 2 reasoning steps (64% of train, 4799 problems)."""
    def __init__(self, subset, split, **kwargs):
        super().__init__(subset, split, min_steps=3, max_steps=float('inf'), **kwargs)


if __name__ == "__main__":
    from collections import Counter
    ds = load_dataset("openai/gsm8k", "main", split="train")
    counts = [_count_steps(row['answer']) for row in ds]
    dist = Counter(counts)
    total = len(counts)
    print(f"\nGSM8K train split: {total} problems")
    print(f"Mean steps: {sum(counts)/total:.1f}")
    print(f"Median steps: {sorted(counts)[total//2]}")
    print(f"\nDistribution:")
    cumulative = 0
    for steps in sorted(dist.keys()):
        n = dist[steps]
        cumulative += n
        print(f"  {steps} steps: {n:>5} problems ({100*n/total:5.1f}%)  cumulative: {cumulative:>5} ({100*cumulative/total:5.1f}%)")
    print(f"\nSuggested splits for even cut:")
    for cutoff in range(1, max(dist.keys())):
        easy = sum(n for s, n in dist.items() if s <= cutoff)
        hard = total - easy
        print(f"  <= {cutoff} / > {cutoff}: {easy} ({100*easy/total:.0f}%) / {hard} ({100*hard/total:.0f}%)")
