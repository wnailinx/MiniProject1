"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    recipe = config.get('training', {})
    schedule_steps = recipe.get('schedule_steps', args.steps)
    peak_lr = recipe.get('learning_rate', .001)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr,
        betas=tuple(recipe.get('betas', (.9, .999))),
        weight_decay=recipe.get('weight_decay', .1),
    )
    ema_decay = recipe.get('ema_decay', 0.)
    ema_start = recipe.get('ema_start', 0)
    ema_model = copy.deepcopy(model).eval() if ema_decay else None
    ema_started = False
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    sampler = recipe.get('sampler', 'random')
    epoch_starts = torch.empty(0, dtype=torch.long)
    epoch_cursor = 0

    def sample_starts():
        nonlocal epoch_starts, epoch_cursor
        if sampler == 'random':
            return torch.randint(len(tokens)-257, (args.batch_size,), generator=rng)
        if sampler != 'epoch':
            raise ValueError(f'Unknown training sampler: {sampler}')
        if epoch_cursor + args.batch_size > len(epoch_starts):
            offset = torch.randint(256, (1,), generator=rng).item()
            epoch_starts = torch.arange(offset, len(tokens)-257, 256)
            epoch_starts = epoch_starts[torch.randperm(len(epoch_starts), generator=rng)]
            epoch_cursor = 0
        result = epoch_starts[epoch_cursor:epoch_cursor + args.batch_size]
        epoch_cursor += args.batch_size
        return result
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    best_validation = None
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = sample_starts().to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        warmup = recipe.get('warmup_steps', 100)
        minimum_ratio = recipe.get('min_lr_ratio', .1)
        if step + 1 <= warmup:
            learning_rate = peak_lr * (step + 1) / warmup
        else:
            progress = min(1., (step + 1 - warmup) / max(1, schedule_steps - warmup))
            learning_rate = peak_lr * (
                minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))
            )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            if hasattr(model, 'training_loss'):
                loss, training_metrics = model.training_loss(batch)
            else:
                loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
                training_metrics = {}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if ema_model is not None and step + 1 >= ema_start:
            if not ema_started:
                ema_model.load_state_dict(model.state_dict())
                ema_started = True
            else:
                with torch.no_grad():
                    for averaged, current in zip(ema_model.parameters(), model.parameters()):
                        averaged.lerp_(current.detach(), 1 - ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'lr':learning_rate,
                   'seconds':time.perf_counter()-started-intermediate_validation_seconds,
                   **{name: value.detach().float().item() for name, value in training_metrics.items()}}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            evaluation_model = ema_model if ema_started else model
            intermediate = score(evaluation_model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if recipe.get('save_best', False) and (
                best_validation is None or intermediate['bpb'] < best_validation['bpb']
            ):
                best_validation = {'step': step + 1, **intermediate}
                torch.save({
                    'protocol': PROTOCOL, 'implementation': args.implementation,
                    'config': config, 'model': evaluation_model.state_dict(),
                    'seed': args.seed,
                    'train_tokens': (step + 1) * args.batch_size * 256,
                }, args.run_dir / 'best_checkpoint.pt')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    evaluation_model = ema_model if ema_started else model
    validation = score(evaluation_model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':evaluation_model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'best_validation':best_validation,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
