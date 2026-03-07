"""
Databricks Dolly 15K dataset.
https://huggingface.co/datasets/databricks/databricks-dolly-15k
"""

from datasets import load_dataset
from tasks.common import Task


class Dolly(Task):
    """
    Databricks Dolly 15K dataset. ~15K rows, train split only.
    Each row has: instruction, context, response, category
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "Dolly only has a train split"
        self.ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        instruction = row["instruction"].strip()
        context = row["context"].strip()
        response = row["response"].strip()

        # If there is additional context, prepend it to the instruction
        # so the user message is self-contained
        if context:
            user_content = f"{instruction}\n\n{context}"
        else:
            user_content = instruction

        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": response},
        ]

        assert len(messages) == 2
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)

        conversation = {"messages": messages}
        return conversation