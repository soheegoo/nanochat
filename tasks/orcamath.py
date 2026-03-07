"""
Microsoft ORCA Math Word Problems dataset.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
"""

from datasets import load_dataset
from tasks.common import Task


class OrcaMath(Task):
    """
    Microsoft ORCA Math Word Problems. ~200K rows, train split only.
    Each row has: question, answer
    """

    def __init__(self, split="train", stop=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "OrcaMath only has a train split"
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train").shuffle(seed=42)
        if stop is not None:
            ds = ds.select(range(min(stop, len(ds))))
        self.ds = ds
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        question = row["question"].strip()
        answer = row["answer"].strip()

        assert question, f"Empty question at index {index}"
        assert answer,   f"Empty answer at index {index}"

        messages = [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]

        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)

        conversation = {"messages": messages}
        return conversation