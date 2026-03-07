"""
AI-MO NuminaMath Chain-of-Thought dataset.
https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
"""

from datasets import load_dataset
from tasks.common import Task


class NuminaMath(Task):
    """
    AI-MO NuminaMath Chain-of-Thought dataset.
    Each row has: problem, solution
    """

    def __init__(self, split="train", stop=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "NuminaMath split must be train|test"
        ds = load_dataset("AI-MO/NuminaMath-CoT", split=split).shuffle(seed=42)
        if stop is not None:
            ds = ds.select(range(min(stop, len(ds))))
        self.ds = ds
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        problem  = row["problem"].strip()
        solution = row["solution"].strip()

        assert problem,  f"Empty problem at index {index}"
        assert solution, f"Empty solution at index {index}"

        messages = [
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ]

        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)

        conversation = {"messages": messages}
        return conversation