"""
GSM8K-AUG task for SFT.
Dataset: https://huggingface.co/datasets/whynlp/gsm8k-aug
386K augmented GSM8K problems with step-by-step calculator expressions.
"""

from datasets import load_dataset

class GSM8KAug:
    def __init__(self, split="train", stop=None):
        assert split in {"train", "validation", "test"}
        ds = load_dataset("whynlp/gsm8k-aug", split=split)
        if stop is not None:
            ds = ds.select(range(min(stop, len(ds))))
        self.data = ds

    def num_examples(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def get_example(self, index):
        row = self.data[index]
        question = row["question"]
        steps = row["steps"]
        answer = row["answer"]

        # Convert steps list into readable solution ending with #### format
        # steps look like: ["<<600*30/100=180>>", "<<600*10/100=60>>", ...]
        solution = " ".join(steps) + f" #### {answer}"

        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution},
            ]
        }

    def __getitem__(self, index):
        return self.get_example(index)