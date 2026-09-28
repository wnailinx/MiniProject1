"""A modernized causal GPT using mechanisms covered in Lecture 3.

The model replaces learned absolute positions, LayerNorm, and the GELU MLP with
RoPE, RMSNorm, and a parameter-matched SwiGLU feed-forward network. Configuration
switches make each change independently ablatable without touching the evaluator.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width, conv_kernel=0):
        super().__init__()
        # Three matrices at about the same parameter cost as a 4x GELU MLP.
        hidden = math.ceil((8 * width / 3) / 8) * 8
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.conv_kernel = conv_kernel
        self.local = (
            nn.Conv1d(hidden, hidden, conv_kernel, groups=hidden, bias=False)
            if conv_kernel > 1 else None
        )
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        value = self.value(x)
        if self.local is not None:
            value = self.local(
                F.pad(value.transpose(1, 2), (self.conv_kernel - 1, 0))
            ).transpose(1, 2)
        return self.out(F.silu(self.gate(x)) * value)


class SparseMoE(nn.Module):
    """Top-1 token-routed SwiGLU experts with Switch-style router losses."""

    def __init__(self, width, experts):
        super().__init__()
        self.num_experts = experts
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(width) for _ in range(experts)])

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat).float()
        probabilities = router_logits.softmax(dim=-1)
        assignments = probabilities.argmax(dim=-1)
        selected = probabilities.gather(1, assignments[:, None]).squeeze(1)

        # Forward scale is one, while the straight-through ratio still trains
        # the selected router probability from the language-model objective.
        gates = (selected / selected.detach().clamp_min(1e-6)).to(flat.dtype)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            locations = (assignments == index).nonzero(as_tuple=False).squeeze(1)
            if locations.numel():
                values = expert(flat.index_select(0, locations))
                values = values * gates.index_select(0, locations)[:, None]
                output.index_copy_(0, locations, values)

        load = F.one_hot(assignments, self.num_experts).float().mean(dim=0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.num_experts * (load.detach() * importance).sum()
        router_z_loss = router_logits.logsumexp(dim=-1).square().mean()
        return output.view(shape), balance_loss, router_z_loss, load.detach()


class HashedContextMemory(nn.Module):
    """A compact learned lookup for causal token n-gram contexts."""

    def __init__(self, width, order, buckets, rank, dropout=0.):
        super().__init__()
        if buckets & (buckets - 1):
            raise ValueError('Hashed-context bucket count must be a power of two.')
        self.order = order
        self.mask = buckets - 1
        self.embedding = nn.Embedding(buckets, rank)
        self.projection = nn.Linear(rank, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.))

    def forward(self, ids):
        length = ids.shape[1]
        padded = F.pad(ids, (self.order - 1, 0), value=2048)
        key = torch.zeros_like(ids)
        # Add one so that real token zero differs from the padding sentinel;
        # order is also encoded by using a separate table.
        for offset in range(self.order):
            token = padded[:, offset:offset + length]
            key = (key * 2049 + token + 1) & self.mask
        memory = self.projection(self.embedding(key))
        return torch.sigmoid(self.gate) * self.dropout(memory)


def _token_byte_table(vocab_size):
    """Decode the fixed ByteLevel-BPE vocabulary into its underlying bytes."""
    tokenizer_path = Path(__file__).resolve().parent / 'data' / 'tokenizer.json'
    tokenizer = json.loads(tokenizer_path.read_text(encoding='utf8'))
    vocabulary = tokenizer['model']['vocab']
    if len(vocabulary) != vocab_size:
        raise ValueError('Tokenizer vocabulary does not match model config.')

    byte_values = list(range(ord('!'), ord('~') + 1))
    byte_values += list(range(161, 173)) + list(range(174, 256))
    unicode_values = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    decoder = {
        chr(character): value
        for value, character in zip(byte_values, unicode_values)
    }

    pieces = [None] * vocab_size
    for token, index in vocabulary.items():
        pieces[index] = [decoder[character] for character in token]
    maximum = max(map(len, pieces))
    table = torch.full((vocab_size, maximum), 256, dtype=torch.long)
    lengths = torch.empty(vocab_size, dtype=torch.long)
    for index, piece in enumerate(pieces):
        lengths[index] = len(piece)
        table[index, :len(piece)] = torch.tensor(piece)
    return table, lengths


class ByteComposition(nn.Module):
    """Share token statistics through the bytes forming each BPE symbol."""

    def __init__(self, vocab, width, feature_width=64, gate_init=-1.38629436):
        super().__init__()
        byte_ids, lengths = _token_byte_table(vocab)
        self.register_buffer('byte_ids', byte_ids, persistent=False)
        self.register_buffer('lengths', lengths, persistent=False)
        self.bytes = nn.Embedding(257, feature_width, padding_idx=256)
        self.length = nn.Embedding(int(lengths.max()) + 1, feature_width)
        self.projection = nn.Linear(4 * feature_width, width, bias=False)
        self.norm = RMSNorm(width)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self):
        features = self.bytes(self.byte_ids)
        mask = self.byte_ids != 256
        mean = (features * mask[..., None]).sum(dim=1) / self.lengths[:, None]
        first = features[:, 0]
        last_index = self.lengths - 1
        last = features[torch.arange(len(features), device=features.device), last_index]
        length = self.length(self.lengths)
        composed = self.norm(self.projection(torch.cat((mean, first, last, length), dim=-1)))
        return torch.sigmoid(self.gate) * .02 * composed


def apply_rope(x, positions, base=10_000.):
    """Rotate pairs in q or k; x has shape [batch, heads, time, head_dim]."""
    half = x.shape[-1] // 2
    frequencies = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class ModernBlock(nn.Module):
    def __init__(
        self, width, heads, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, dropout=0., moe_experts=1, conv_kernel=0,
    ):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError('width / heads must be an even integer for RoPE')
        self.heads = heads
        self.use_rope = use_rope
        self.dropout = dropout
        self.residual_dropout = nn.Dropout(dropout)
        norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1, self.norm2 = norm(width), norm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        if moe_experts > 1:
            if not use_swiglu:
                raise ValueError('Sparse experts require SwiGLU.')
            self.mlp = SparseMoE(width, moe_experts)
        else:
            self.mlp = SwiGLU(width, conv_kernel=conv_kernel) if use_swiglu else nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
            )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            positions = torch.arange(length, device=x.device)
            q, k = apply_rope(q, positions), apply_rope(k, positions)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=True
        )
        x = x + self.residual_dropout(
            self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        )
        normalized = self.norm2(x)
        if isinstance(self.mlp, SparseMoE):
            mlp_output, balance, router_z, load = self.mlp(normalized)
        else:
            mlp_output = self.mlp(normalized)
            balance = router_z = x.new_zeros(())
            load = None
        return x + self.residual_dropout(mlp_output), balance, router_z, load


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, depth = config['width'], config['depth']
        self.use_rope = config.get('use_rope', True)
        self.mtp_steps = config.get('mtp_steps', 1)
        self.mtp_weight = config.get('mtp_weight', .2)
        self.mtp_decay = config.get('mtp_decay', .5)
        self.moe_balance_weight = config.get('moe_balance_weight', .01)
        self.router_z_weight = config.get('router_z_weight', .001)
        self.mos_components = config.get('mos_components', 1)
        self.token_dropout = config.get('token_dropout', 0.)
        self.label_smoothing = config.get('label_smoothing', 0.)
        self.rdrop_alpha = config.get('rdrop_alpha', 0.)
        self.suffix_cache_weights = config.get('suffix_cache_weights')
        self.ngram_weight = config.get('ngram_weight', config.get('trigram_weight', 0.))
        self.token = nn.Embedding(config['vocab'], width)
        byte_config = config.get('byte_composition')
        self.byte_composition = ByteComposition(
            config['vocab'], width,
            feature_width=byte_config.get('feature_width', 64),
            gate_init=byte_config.get('gate_init', -1.38629436),
        ) if byte_config else None
        self.pos = None if self.use_rope else nn.Embedding(self.context, width)
        self.embedding_dropout = nn.Dropout(config.get('dropout', 0.))
        moe_layers = config.get('moe_layers', [])
        if moe_layers == 'all':
            moe_layers = range(depth)
        moe_layers = set(moe_layers)
        self.blocks = nn.ModuleList([
            ModernBlock(
                width, config['heads'],
                use_rmsnorm=config.get('use_rmsnorm', True),
                use_swiglu=config.get('use_swiglu', True),
                use_rope=self.use_rope,
                dropout=config.get('dropout', 0.),
                moe_experts=config.get('moe_experts', 1) if index in moe_layers else 1,
                conv_kernel=config.get('conv_kernel', 0),
            ) for index in range(depth)
        ])
        norm = RMSNorm if config.get('use_rmsnorm', True) else nn.LayerNorm
        self.norm = norm(width)
        self.context_memories = nn.ModuleList([
            HashedContextMemory(
                width, specification['order'], specification['buckets'],
                specification.get('rank', 32), config.get('memory_dropout', .1),
            )
            for specification in config.get('hash_memories', [])
        ])
        self.memory_norm = norm(width) if self.context_memories else nn.Identity()
        self.head = nn.Linear(width, config['vocab'], bias=config.get('output_bias', False))
        if self.mos_components > 1:
            self.mos_latent = nn.Linear(width, self.mos_components * width)
            self.mos_prior = nn.Linear(width, self.mos_components)
        self.mtp_projections = nn.ModuleList([
            nn.Linear(width, width, bias=False) for _ in range(self.mtp_steps - 1)
        ])
        self.apply(self.initialize)
        if self.byte_composition is not None:
            with torch.no_grad():
                self.byte_composition.bytes.weight[256].zero_()
        for memory in self.context_memories:
            nn.init.normal_(memory.embedding.weight, std=.1)
            nn.init.normal_(memory.projection.weight, std=.05)
        # Scale residual-output matrices as depth grows (GPT-2 initialization).
        residual_std = .02 / math.sqrt(2 * depth)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            if isinstance(block.mlp, SparseMoE):
                for expert in block.mlp.experts:
                    nn.init.normal_(expert.out.weight, std=residual_std)
            else:
                output = block.mlp.out if isinstance(block.mlp, SwiGLU) else block.mlp[-1]
                nn.init.normal_(output.weight, std=residual_std)
                if isinstance(block.mlp, SwiGLU) and block.mlp.local is not None:
                    nn.init.zeros_(block.mlp.local.weight)
                    block.mlp.local.weight.data[:, 0, -1] = 1.
        for projection in self.mtp_projections:
            nn.init.eye_(projection.weight)
        if self.mos_components > 1:
            nn.init.zeros_(self.mos_prior.weight)
            nn.init.zeros_(self.mos_prior.bias)
            nn.init.zeros_(self.mos_latent.bias)
            with torch.no_grad():
                for component in range(self.mos_components):
                    block = self.mos_latent.weight[component * width:(component + 1) * width]
                    nn.init.eye_(block)
                    block.add_(torch.randn_like(block) * .001)
        self.head.weight = self.token.weight
        self._unfolded_token_weight = None
        trigram = config.get('trigram_assets')
        if trigram:
            self.register_buffer(
                'ngram_bigram_probs',
                torch.zeros(config['vocab'], config['vocab'], dtype=torch.float16),
            )
            self.register_buffer(
                'ngram_context_keys',
                torch.zeros(trigram['contexts'], dtype=torch.int64),
            )
            self.register_buffer(
                'ngram_offsets',
                torch.zeros(trigram['contexts'] + 1, dtype=torch.int32),
            )
            self.register_buffer(
                'ngram_targets',
                torch.zeros(trigram['entries'], dtype=torch.int16),
            )
            self.register_buffer(
                'ngram_edge_probs',
                torch.zeros(trigram['entries'], dtype=torch.float16),
            )
            self.register_buffer(
                'ngram_backoff',
                torch.ones(trigram['contexts'], dtype=torch.float16),
            )
        else:
            self.ngram_bigram_probs = None
        fourgram = config.get('fourgram_assets')
        if fourgram:
            self.register_buffer(
                'fourgram_context_keys',
                torch.zeros(fourgram['contexts'], dtype=torch.int64),
            )
            self.register_buffer(
                'fourgram_offsets',
                torch.zeros(fourgram['contexts'] + 1, dtype=torch.int32),
            )
            self.register_buffer(
                'fourgram_targets',
                torch.zeros(fourgram['entries'], dtype=torch.int16),
            )
            self.register_buffer(
                'fourgram_edge_probs',
                torch.zeros(fourgram['entries'], dtype=torch.float16),
            )
            self.register_buffer(
                'fourgram_backoff',
                torch.ones(fourgram['contexts'], dtype=torch.float16),
            )
        else:
            self.fourgram_context_keys = None
        fivegram = config.get('fivegram_assets')
        if fivegram:
            self.register_buffer(
                'fivegram_context_keys',
                torch.zeros(fivegram['contexts'], dtype=torch.int64),
            )
            self.register_buffer(
                'fivegram_offsets',
                torch.zeros(fivegram['contexts'] + 1, dtype=torch.int32),
            )
            self.register_buffer(
                'fivegram_targets',
                torch.zeros(fivegram['entries'], dtype=torch.int16),
            )
            self.register_buffer(
                'fivegram_edge_probs',
                torch.zeros(fivegram['entries'], dtype=torch.float16),
            )
            self.register_buffer(
                'fivegram_backoff',
                torch.ones(fivegram['contexts'], dtype=torch.float16),
            )
        else:
            self.fivegram_context_keys = None

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def embedding_weights(self):
        weights = self.token.weight
        if self.byte_composition is not None:
            weights = weights + self.byte_composition()
        return weights

    def train(self, mode=True):
        # At inference, fold the deterministic byte contribution into the tied
        # embedding/head Parameter. This preserves PyTorch's optimized CPU
        # embedding and Linear paths. Restore the learned direct table before
        # resuming training or saving a training checkpoint.
        if mode and self._unfolded_token_weight is not None:
            with torch.no_grad():
                self.token.weight.copy_(self._unfolded_token_weight)
            self._unfolded_token_weight = None
        result = super().train(mode)
        if not mode and self.byte_composition is not None and self._unfolded_token_weight is None:
            with torch.no_grad():
                self._unfolded_token_weight = self.token.weight.detach().clone()
                self.token.weight.add_(self.byte_composition())
        return result

    def output_logits(self, hidden, token_weights=None):
        if token_weights is None:
            return self.head(hidden)
        return F.linear(hidden, token_weights, self.head.bias)

    def features(self, ids, return_aux=False, token_weights=None):
        x = self.token(ids) if token_weights is None else F.embedding(ids, token_weights)
        if self.training and self.token_dropout:
            drop = torch.rand((*ids.shape, 1), device=ids.device) < self.token_dropout
            x = x.masked_fill(drop, 0.)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.embedding_dropout(x)
        balances, router_z_losses, loads = [], [], []
        for block in self.blocks:
            x, balance, router_z, load = block(x)
            if load is not None:
                balances.append(balance)
                router_z_losses.append(router_z)
                loads.append(load)
        x = self.norm(x)
        if self.context_memories:
            memory = torch.stack([table(ids) for table in self.context_memories]).sum(dim=0)
            x = self.memory_norm(x + memory)
        if return_aux:
            zero = x.new_zeros(())
            auxiliary = {
                'moe_balance': torch.stack(balances).mean() if balances else zero,
                'router_z': torch.stack(router_z_losses).mean() if router_z_losses else zero,
                'expert_load': torch.stack(loads).mean(dim=0) if loads else None,
            }
            return x, auxiliary
        return x

    def forward(self, ids):
        token_weights = (
            self.embedding_weights()
            if self.byte_composition is not None and self._unfolded_token_weight is None
            else None
        )
        hidden = self.features(ids, token_weights=token_weights)
        if self.mos_components > 1:
            return self.log_probs_from_hidden(
                hidden, self.token.weight if token_weights is None else token_weights,
            )
        return self.output_logits(hidden, token_weights)

    def predict_log_probs(self, ids):
        output = self(ids)
        if self.mos_components > 1:
            log_probs = output
        else:
            log_probs = F.log_softmax(output.float(), dim=-1)
        if self.suffix_cache_weights:
            log_probs = self.mix_suffix_cache(ids, log_probs)
        if self.ngram_weight:
            log_probs = self.mix_static_ngram(ids, log_probs)
        return log_probs

    @staticmethod
    def apply_sparse_backoff(
        contexts, backoff, context_keys, offsets, targets,
        edge_probs, backoff_weights,
    ):
        """Apply one CSR absolute-discount order to dense backoff rows."""
        vocab = backoff.shape[-1]
        rows = torch.searchsorted(context_keys, contexts)
        valid = rows < len(context_keys)
        safe_rows = rows.clamp_max(len(context_keys) - 1)
        valid &= context_keys[safe_rows] == contexts
        starts = torch.zeros_like(rows, dtype=torch.int64)
        ends = torch.zeros_like(rows, dtype=torch.int64)
        starts[valid] = offsets[safe_rows[valid]].long()
        ends[valid] = offsets[safe_rows[valid] + 1].long()
        counts = ends - starts

        multiplier = torch.ones(len(rows), device=contexts.device, dtype=torch.float32)
        multiplier[valid] = backoff_weights[safe_rows[valid]].float()
        ngram_probs = backoff * multiplier[:, None]
        total_edges = int(counts.sum().item())
        if total_edges:
            output_rows = torch.repeat_interleave(
                torch.arange(len(rows), device=contexts.device), counts,
            )
            segment_starts = counts.cumsum(0) - counts
            within_segment = torch.arange(total_edges, device=contexts.device)
            within_segment -= torch.repeat_interleave(segment_starts, counts)
            edge_indices = torch.repeat_interleave(starts, counts) + within_segment
            edge_targets = targets[edge_indices].long()
            values = edge_probs[edge_indices].float()
            flat_indices = output_rows * vocab + edge_targets
            ngram_probs.view(-1).scatter_add_(0, flat_indices, values)
        ngram_probs /= ngram_probs.sum(dim=-1, keepdim=True)
        return ngram_probs

    def mix_static_ngram(self, ids, neural_log_probs):
        """Mix a training-derived, window-causal backoff n-gram model."""
        batch, length = ids.shape
        vocab = neural_log_probs.shape[-1]
        bigram = self.ngram_bigram_probs[ids].float().flatten(0, 1)
        trigram_contexts = torch.full_like(ids, -1)
        trigram_contexts[:, 1:] = ids[:, :-1] * vocab + ids[:, 1:]
        ngram_probs = self.apply_sparse_backoff(
            trigram_contexts.flatten(), bigram,
            self.ngram_context_keys, self.ngram_offsets,
            self.ngram_targets, self.ngram_edge_probs, self.ngram_backoff,
        )
        if self.fourgram_context_keys is not None:
            fourgram_contexts = torch.full_like(ids, -1)
            fourgram_contexts[:, 2:] = (
                (ids[:, :-2] * vocab + ids[:, 1:-1]) * vocab + ids[:, 2:]
            )
            ngram_probs = self.apply_sparse_backoff(
                fourgram_contexts.flatten(), ngram_probs,
                self.fourgram_context_keys, self.fourgram_offsets,
                self.fourgram_targets, self.fourgram_edge_probs,
                self.fourgram_backoff,
            )
        if self.fivegram_context_keys is not None:
            fivegram_contexts = torch.full_like(ids, -1)
            fivegram_contexts[:, 3:] = (
                (
                    (ids[:, :-3] * vocab + ids[:, 1:-2]) * vocab
                    + ids[:, 2:-1]
                ) * vocab + ids[:, 3:]
            )
            ngram_probs = self.apply_sparse_backoff(
                fivegram_contexts.flatten(), ngram_probs,
                self.fivegram_context_keys, self.fivegram_offsets,
                self.fivegram_targets, self.fivegram_edge_probs,
                self.fivegram_backoff,
            )
        ngram_log_probs = ngram_probs.clamp_min(1e-12).log().view_as(neural_log_probs)
        weight = self.ngram_weight
        return torch.logaddexp(
            neural_log_probs + math.log1p(-weight),
            ngram_log_probs + math.log(weight),
        )

    def mix_suffix_cache(self, ids, neural_log_probs):
        """Mix exact earlier continuations from the current causal window."""
        batch, length = ids.shape
        equal = ids[:, :, None] == ids[:, None, :]
        matches = torch.zeros(
            batch, length, length, dtype=torch.int16, device=ids.device,
        )
        for position in range(length):
            matches[:, position, 0] = equal[:, position, 0]
            if position:
                matches[:, position, 1:] = (
                    matches[:, position - 1, :-1] + 1
                ) * equal[:, position, 1:]
        positions = torch.arange(length, device=ids.device)
        causal = positions[None, :] < positions[:, None]
        matches.masked_fill_(~causal[None], 0)
        longest = matches.max(dim=-1).values
        selected = (matches == longest[:, :, None]) & (longest[:, :, None] > 0)

        # A match ending at i votes for x[i+1], which is already observed since
        # the causal mask requires i < the prediction position.
        continuation = F.pad(ids[:, 1:], (0, 1))
        counts = torch.zeros_like(neural_log_probs)
        counts.scatter_add_(
            2,
            continuation[:, None, :].expand(batch, length, length),
            selected.to(counts.dtype),
        )
        counts /= selected.sum(dim=-1, keepdim=True).clamp_min(1)

        table = neural_log_probs.new_tensor((0., *self.suffix_cache_weights))
        weight = table[longest.long().clamp_max(len(table) - 1)].unsqueeze(-1)
        copy_log_probs = torch.where(
            counts > 0, counts.clamp_min(1e-12).log(), -torch.inf,
        )
        return torch.logaddexp(
            neural_log_probs + torch.log1p(-weight),
            copy_log_probs + weight.clamp_min(1e-12).log(),
        )

    def log_probs_from_hidden(self, hidden, token_weights=None):
        if token_weights is None:
            token_weights = self.embedding_weights()
        batch, length, width = hidden.shape
        latent = torch.tanh(self.mos_latent(hidden)).view(
            batch, length, self.mos_components, width
        )
        component_logits = F.linear(latent, token_weights)
        component_log_probs = F.log_softmax(component_logits.float(), dim=-1)
        mixture_log_probs = F.log_softmax(self.mos_prior(hidden).float(), dim=-1)
        return torch.logsumexp(
            component_log_probs + mixture_log_probs.unsqueeze(-1), dim=2
        )

    def training_loss(self, tokens):
        """Causal next-token loss plus training-only future-token supervision."""
        if self.rdrop_alpha:
            return self.rdrop_training_loss(tokens)
        token_weights = self.embedding_weights()
        hidden, auxiliary = self.features(
            tokens[:, :-1], return_aux=True, token_weights=token_weights,
        )
        if self.mos_components > 1:
            log_probs = self.log_probs_from_hidden(hidden, token_weights).flatten(0, 1)
            targets = tokens[:, 1:].flatten()
            main_loss = -(
                (1 - self.label_smoothing)
                * log_probs.gather(1, targets[:, None]).squeeze(1)
                + self.label_smoothing * log_probs.mean(dim=-1)
            ).mean()
        else:
            main_loss = F.cross_entropy(
                self.output_logits(hidden, token_weights).flatten(0, 1), tokens[:, 1:].flatten(),
                label_smoothing=self.label_smoothing,
            )
        loss = main_loss
        metrics = {'main_loss': main_loss.detach()}
        for offset, projection in enumerate(self.mtp_projections, start=2):
            length = tokens.shape[1] - offset
            logits = self.output_logits(projection(hidden[:, :length]), token_weights)
            future_loss = F.cross_entropy(
                logits.flatten(0, 1), tokens[:, offset:offset + length].flatten()
            )
            weight = self.mtp_weight * self.mtp_decay ** (offset - 2)
            loss = loss + weight * future_loss
            metrics[f'mtp_{offset}_loss'] = future_loss.detach()
        loss = loss + self.moe_balance_weight * auxiliary['moe_balance']
        loss = loss + self.router_z_weight * auxiliary['router_z']
        metrics['moe_balance'] = auxiliary['moe_balance'].detach()
        metrics['router_z'] = auxiliary['router_z'].detach()
        if auxiliary['expert_load'] is not None:
            for index, value in enumerate(auxiliary['expert_load']):
                metrics[f'expert_{index}_load'] = value
        return loss, metrics

    def rdrop_training_loss(self, tokens):
        if self.mtp_projections:
            raise ValueError('R-Drop and multi-token heads are separate ablations.')
        targets = tokens[:, 1:].flatten()
        token_weights = self.embedding_weights()
        hidden_a, auxiliary_a = self.features(
            tokens[:, :-1], return_aux=True, token_weights=token_weights,
        )
        hidden_b, auxiliary_b = self.features(
            tokens[:, :-1], return_aux=True, token_weights=token_weights,
        )
        if self.mos_components > 1:
            log_a = self.log_probs_from_hidden(hidden_a, token_weights).flatten(0, 1)
            log_b = self.log_probs_from_hidden(hidden_b, token_weights).flatten(0, 1)
            def supervised(log_probs):
                return -(
                    (1 - self.label_smoothing)
                    * log_probs.gather(1, targets[:, None]).squeeze(1)
                    + self.label_smoothing * log_probs.mean(dim=-1)
                ).mean()
        else:
            logits_a = self.output_logits(hidden_a, token_weights).flatten(0, 1).float()
            logits_b = self.output_logits(hidden_b, token_weights).flatten(0, 1).float()
            log_a, log_b = logits_a.log_softmax(-1), logits_b.log_softmax(-1)
            def supervised(log_probs):
                return -(
                    (1 - self.label_smoothing)
                    * log_probs.gather(1, targets[:, None]).squeeze(1)
                    + self.label_smoothing * log_probs.mean(dim=-1)
                ).mean()
        supervised_loss = .5 * (supervised(log_a) + supervised(log_b))
        symmetric_kl = .25 * (
            (log_a.exp() * (log_a - log_b)).sum(dim=-1).mean()
            + (log_b.exp() * (log_b - log_a)).sum(dim=-1).mean()
        )
        balance = .5 * (auxiliary_a['moe_balance'] + auxiliary_b['moe_balance'])
        router_z = .5 * (auxiliary_a['router_z'] + auxiliary_b['router_z'])
        loss = (
            supervised_loss + self.rdrop_alpha * symmetric_kl
            + self.moe_balance_weight * balance + self.router_z_weight * router_z
        )
        return loss, {
            'main_loss': supervised_loss.detach(),
            'rdrop_kl': symmetric_kl.detach(),
            'moe_balance': balance.detach(),
            'router_z': router_z.detach(),
        }


def build_model(config):
    return ModernGPT(config)
