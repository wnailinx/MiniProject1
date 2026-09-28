# DASE7506 Mini-Project 1 submission

This directory is the self-contained submission candidate for the frozen model.
It contains the exact evaluation code, fixed tokenizer and data, one final
checkpoint, the report source and recorded CPU FP32 results.

## Frozen result

| Item | Value |
|---|---:|
| Validation BPB | 1.4602616594 |
| Full-test BPB | **1.4828487177** |
| Full-test scoring time | 37.71 s |
| Paired baseline test time | 10.58 s |
| Test scoring-time ratio | 3.56× |
| Peak CPU working set | 1.833 GiB |
| Checkpoint size | 62,212,573 bytes (59.33 MiB) |
| Checkpoint SHA256 | `3e29029d5760d84e5a062bd6acddd57e1342a83844c15866d5d0e09b5c4c8100` |

The final route was selected using validation data. A separately frozen dense
candidate had been evaluated on test before this route began; its score was not
used to choose the final architecture, checkpoint or mixture settings. The final
checkpoint was scored after it was frozen, and its test result was not used for
further tuning.

## Directory layout

- `checkpoint.pt`: the only submitted inference checkpoint.
- `checkpoint_lineage.json`: source checkpoint and training-only n-gram lineage.
- `code/`: model, trainer, fixed evaluator, dependencies, tests and supplied data.
- `results/`: recorded CPU FP32 validation and test summaries.
- `REPORT.md`: report source; fill the remaining identity and repository fields.
- `COMPLIANCE_AUDIT.md`: constraint-by-constraint audit and measured evidence.
- `SUBMISSION_MANIFEST.json`: SHA256 and size of every submitted file except itself.

## Install

Python 3.12 and PyTorch 2.7.1 are the tested versions. For CPU reproduction:

```powershell
conda create -n 7506_1 python=3.12 -y
conda activate 7506_1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements.txt
```

For training on a compatible NVIDIA GPU, install the CUDA 12.6 build instead:

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r code/requirements.txt
```

No API key, network access, external dataset or pretrained weight is needed after
dependency installation.

## Reproduce the submitted score

Run these commands from `MP1_submit/code/`:

```powershell
python -m unittest discover -s tests -v
python evaluate.py `
  --checkpoint ../checkpoint.pt `
  --device cpu --precision fp32 --threads 4 `
  --split test --output ../reproduced_test_cpu_fp32.json
```

The expected `bpb` is `1.4828487177100396`. Small timing differences across CPUs
are expected. The evaluator output records the checkpoint, implementation,
evaluator and tokenizer hashes.

To verify the checkpoint before evaluation:

```powershell
Get-FileHash ../checkpoint.pt -Algorithm SHA256
```

## Rebuild from training data

Direct evaluation uses the included checkpoint and does not require retraining.
The neural source can be reproduced on an NVIDIA GPU from `MP1_submit/code/`:

```powershell
python train.py --implementation student `
  --config configs/moe-mtp-original-7x224.json `
  --device cuda --precision bf16 --threads 4 --seed 17 `
  --steps 12000 --batch-size 32 --eval-every 600 `
  --run-dir runs/moe-mtp-original-12000

python build_ngram_checkpoint.py `
  --source runs/moe-mtp-original-12000/best_checkpoint.pt `
  --output runs/moe-mtp-final/checkpoint.pt
```

The selected neural run processed 98,304,000 targets and used 1,341.55 GPU
training seconds on an NVIDIA GeForce RTX 4070 Laptop GPU. The n-gram builder
reads only `data/wikitext_train.txt` and performs no gradient training.

## Reused work and AI assistance

The implementation builds on the course-provided GPT, training pipeline,
tokenizer, data loader and evaluator. PyTorch's scaled dot-product attention and
optimization primitives are used directly. No external training text or
pretrained weights are used.

OpenAI Codex was used to help interpret the assignment, design and implement
model variants, check causality and resource limits, run and interpret ablations,
optimize the sparse n-gram path, and draft documentation.

## Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher in *Pointer Sentinel Mixture Models*. The text is by Wikipedia
contributors. The upstream dataset identifies CC BY-SA 3.0 and the GNU Free
Documentation License. Dataset hashes and the fixed revision are recorded in
`code/data/manifest.json` and `code/README.md`.
