## Chat RL
timestamp: 2026-03-08 17:00:47

- run: nanochat-baseline-mar-8-maryam
- device_type: 
- dtype: bfloat16
- model_tag: nanochat-baseline-mar-8-maryam
- model_step: 500
- num_epochs: 1
- device_batch_size: 8
- examples_per_step: 16
- num_samples: 16
- max_new_tokens: 256
- temperature: 1.0000
- top_k: 50
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0500
- eval_every: 60
- eval_examples: 400
- save_every: 60

