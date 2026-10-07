"""Selected winner loading and source provenance for future runs."""
from pathlib import Path
import torch
from torch import nn
from tokenizers import Tokenizer
from settings import ModelConfig
from storage import digest
from models.branch_sigmoid.transformer import Model

WINNER = 'dump/ffn-final-10000-s461/branch_sigmoid/last.pt'
WINNER_SHA256 = '914580a04e9d4b4db2512ace35f917065c98cee21df0353fcc1b187669208954'

class Adapter(nn.Module):
    encoded = False
    def __init__(self, recipe):
        super().__init__()
        self.core = Model(ModelConfig(**recipe['model']), seed=recipe['training']['seed'])
        self.config = self.core.config
    @property
    def blocks(self): return self.core.blocks
    def forward(self, tokens, past=None, mask=None): return self.core(tokens)

def build(recipe, kind='branch_sigmoid'):
    if kind != 'branch_sigmoid': raise ValueError('Only Branch Sigmoid is retained')
    return Adapter(recipe)

def sources():
    paths = list(Path('src/models/branch_sigmoid').glob('*.py'))
    paths += [Path(p) for p in ('src/settings.py','src/storage.py','src/models/components.py',
                              'src/config/branch_sigmoid.json','src/config/branch_data_identity.json')]
    return {p.as_posix():digest(p) for p in sorted(paths)}

def load_winner(device='cpu'):
    if digest(WINNER) != WINNER_SHA256: raise ValueError('Selected checkpoint identity changed')
    state = torch.load(WINNER, weights_only=True, map_location='cpu')
    if state['kind'] != 'branch_sigmoid' or state['total_steps'] != 10000:
        raise ValueError('Unexpected winner checkpoint')
    model = build(state['recipe'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), Tokenizer.from_str(state['tokenizer'])
