"""Step1/2 token CE objectives. Step3 uses models.token_memory."""
from dataclasses import asdict
import torch
from torch.nn import functional as F

class Task:

    def __init__(self, recipe, device):
        self.recipe, self.device = (recipe, device)
        self.kind = recipe['model']
        c, seed = (recipe['config'], recipe['seed'])
        self.codec = None
        if self.kind == 'branch':
            from settings import ModelConfig
            from models.branch_sigmoid.transformer import Model
            self.model = Model(ModelConfig(**c), seed).to(device)
        else:
            from models.position_compressor.residual import Config, Model
            self.model = Model(Config(**c), seed).to(device)

    def examples(self, rows):
        context = self.recipe['config'].get('context', self.recipe['config'].get('max_tokens'))
        result = []
        for row in rows:
            tokens = row['tokens'].flatten()
            if self.kind == 'branch':
                for i in range(0, len(tokens) - 1, context):
                    x = tokens[i:i + context + 1]
                    if len(x) > 1:
                        result.append(x)
            else:
                result.extend((tokens[i:i + context] for i in range(0, len(tokens), context)))
        return result

    def count(self, items):
        return sum((len(t) - (self.kind == 'branch') for t in items))

    def loss(self, items):
        n = max((len(t) for t in items)) - (self.kind == 'branch')
        tokens = torch.zeros(len(items), n, dtype=torch.long, device=self.device)
        targets = torch.full_like(tokens, -100)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, t in enumerate(items):
            x, y = (t[:-1], t[1:]) if self.kind == 'branch' else (t, t)
            tokens[i, :len(x)] = x.to(self.device)
            targets[i, :len(y)] = y.to(self.device)
            mask[i, :len(y)] = True
        logits = self.model(tokens) if self.kind == 'branch' else self.model(tokens, mask)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        with torch.no_grad():
            correct = (logits.argmax(-1) == targets) & mask
            stats = {'tokens': int(mask.sum()), 'correct': int(correct.sum()), 'exact': int((correct | ~mask).all(-1).sum()), 'sequences': len(targets)}
        return (ce, stats)

    @torch.no_grad()
    def evaluate(self, examples, microbatch):
        self.model.eval()
        totals = dict(tokens=0, correct=0, exact=0, sequences=0)
        loss_sum = 0.0
        for i in range(0, len(examples), microbatch):
            loss, stats = self.loss(examples[i:i + microbatch])
            loss_sum += float(loss) * stats['tokens']
            for k in totals:
                totals[k] += stats[k]
        self.model.train()
        return {'ce': loss_sum / totals['tokens'], 'token_accuracy': totals['correct'] / totals['tokens'], **totals}

def config_for(kind, args):

    def value(name, default):
        v = getattr(args, name, None)
        return default if v is None else v
    if kind == 'branch':
        from settings import ModelConfig
        return asdict(ModelConfig(name='branch_sigmoid', width=value('width', 384), heads=value('heads', 6), layers=value('layers', 4), context=value('context', 512), hidden=1024, ffn='branch_sigmoid'))
    from models.position_compressor.residual import Config
    return asdict(Config(width=value('width', 256), heads=value('heads', 4), encoder_layers=value('layers', 4), decoder_layers=value('decoder_layers', 1), max_tokens=value('context', 512), span=64))
