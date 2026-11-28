import torch
from torch import nn
from rl.banana_hog_retrain import StableHogPolicy

class PublicPolicy(StableHogPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        m = self.model
        if (not m.card_conditioned_placement or m.placement_mode != 'heatmap'
                or m.placement_card_mode != 'selected' or getattr(m, 'arena_memory_channels', 0)):
            raise ValueError('Cached scorer supports only the selected-card v4.4.2 architecture')
        self.public_adapter = nn.Sequential(nn.Linear(58, 32), nn.Tanh(), nn.Linear(32, 160))
        nn.init.zeros_(self.public_adapter[-1].weight)
        nn.init.zeros_(self.public_adapter[-1].bias)
        self.model.zone_head.requires_grad_(True)
        self.model.tile_head.requires_grad_(True)
        self.eval()

CARDS = ('hog-rider','musketeer','ice-golem','skeletons','ice-spirit','cannon','fireball','the-log')
DIMENSION = 22


class ContextWait(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base
        self.residual=None

    def forward(self, fused):
        result=self.base(fused)
        if self.residual is None or len(self.residual)!=len(fused):
            raise RuntimeError('Waiting decision is missing its current public state')
        return result+self.residual[:,None]


class SearchPolicy(PublicPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_grad_(False)
        self.card_adjustments = nn.Parameter(torch.zeros(DIMENSION), requires_grad=False)
        self.card_ids = tuple(self.vocab.encode(c) for c in CARDS)
        self.wait_head=ContextWait(self.wait_head)

    def set_vector(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.shape != (DIMENSION,) or not torch.isfinite(values).all():
            raise ValueError(f'Expected {DIMENSION} finite policy coefficients')
        self.card_adjustments.copy_(values)
