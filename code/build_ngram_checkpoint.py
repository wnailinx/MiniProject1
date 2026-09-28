"""Attach compact training-only backoff statistics to a trained MoE checkpoint.

The resulting checkpoint stores FP16 inference assets, while ``evaluate.py``
reconstructs the model's parameters in FP32. All n-gram counts come from the
fixed supplied training split.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from common import load_data


VOCAB = 2048
DISCOUNT = .75


def encode(tokens, order):
    size = len(tokens) - order + 1
    result = np.zeros(size, dtype=np.int64)
    for offset in range(order):
        result = result * VOCAB + tokens[offset:offset + size]
    return result


def grouped(values):
    keys, starts = np.unique(values, return_index=True)
    ends = np.r_[starts[1:], len(values)]
    return keys, starts, ends


def build_bigram(tokens):
    unigram = np.bincount(tokens, minlength=VOCAB).astype(np.float64) + .1
    unigram /= unigram.sum()
    keys, counts = np.unique(encode(tokens, 2), return_counts=True)
    contexts = keys // VOCAB
    targets = keys % VOCAB
    context_keys, starts, ends = grouped(contexts)
    totals = np.add.reduceat(counts.astype(np.float64), starts)
    types = (ends - starts).astype(np.float64)
    probabilities = np.broadcast_to(unigram, (VOCAB, VOCAB)).copy()
    probabilities[context_keys] *= (DISCOUNT * types / totals)[:, None]
    row = np.searchsorted(context_keys, contexts)
    probabilities[contexts, targets] += np.maximum(counts - DISCOUNT, 0) / totals[row]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return torch.from_numpy(probabilities.astype(np.float16))


def build_sparse_order(tokens, order, minimum_count):
    keys, counts = np.unique(encode(tokens, order), return_counts=True)
    contexts = keys // VOCAB
    targets = keys % VOCAB
    context_keys, starts, ends = grouped(contexts)
    totals = np.add.reduceat(counts.astype(np.float64), starts)
    types = (ends - starts).astype(np.float64)
    keep_context = totals >= minimum_count
    keep_entries = np.repeat(keep_context, ends - starts)
    context_keys = context_keys[keep_context]
    totals = totals[keep_context]
    types = types[keep_context]
    lengths = (ends - starts)[keep_context]
    entry_counts = counts[keep_entries]
    entry_targets = targets[keep_entries]
    row = np.repeat(np.arange(len(context_keys)), lengths)
    edge_probs = np.maximum(entry_counts - DISCOUNT, 0) / totals[row]
    backoff = DISCOUNT * types / totals
    return {
        'context_keys': torch.from_numpy(context_keys.astype(np.int64)),
        'offsets': torch.from_numpy(np.r_[0, np.cumsum(lengths)].astype(np.int32)),
        'targets': torch.from_numpy(entry_targets.astype(np.int16)),
        'edge_probs': torch.from_numpy(edge_probs.astype(np.float16)),
        'backoff': torch.from_numpy(backoff.astype(np.float16)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().parent / 'configs/moe-mtp-final-7x224.json',
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    config = json.loads(args.config.read_text())
    tokens = load_data()['train'][0].numpy().astype(np.int64)
    orders = {
        3: build_sparse_order(tokens, 3, 1),
        4: build_sparse_order(tokens, 4, 2),
        5: build_sparse_order(tokens, 5, 3),
    }
    expected = {
        3: config['trigram_assets'],
        4: config['fourgram_assets'],
        5: config['fivegram_assets'],
    }
    for order, assets in orders.items():
        if (
            len(assets['context_keys']) != expected[order]['contexts']
            or len(assets['targets']) != expected[order]['entries']
        ):
            raise ValueError(f'Unexpected order-{order} training statistics.')

    checkpoint = copy.deepcopy(checkpoint)
    checkpoint['config'] = config
    checkpoint['model'] = {
        name: value.half() if value.is_floating_point() else value
        for name, value in checkpoint['model'].items()
    }
    checkpoint['model']['ngram_bigram_probs'] = build_bigram(tokens)
    for order, prefix in ((3, 'ngram'), (4, 'fourgram'), (5, 'fivegram')):
        assets = orders[order]
        checkpoint['model'].update({
            f'{prefix}_context_keys': assets['context_keys'],
            f'{prefix}_offsets': assets['offsets'],
            f'{prefix}_targets': assets['targets'],
            f'{prefix}_edge_probs': assets['edge_probs'],
            f'{prefix}_backoff': assets['backoff'],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    lineage = {
        'source': str(args.source),
        'source_train_tokens': checkpoint['train_tokens'],
        'selection_split': 'validation',
        'new_train_tokens': 0,
        'discount': DISCOUNT,
        'ngram_weight': config['ngram_weight'],
        'minimum_context_counts': {'3': 1, '4': 2, '5': 3},
    }
    args.output.with_name('lineage.json').write_text(json.dumps(lineage, indent=2) + '\n')
    print(json.dumps({
        'checkpoint': str(args.output),
        'checkpoint_mib': args.output.stat().st_size / 2**20,
        **{f'order_{order}_contexts': len(assets['context_keys'])
           for order, assets in orders.items()},
    }, indent=2))


if __name__ == '__main__':
    main()
