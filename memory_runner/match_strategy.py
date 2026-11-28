"""Configuration and legal tiles for the full-match policy."""
from dataclasses import dataclass
import json
import numpy as np

from defensive_placement import X, Y
from defense_responses import can_hit
from live_decision import Move


@dataclass(frozen=True)
class MatchStrategy:
    rollout_weight: float = 1.
    rollout_threshold: float = 0.
    rollout_interval: int = 20
    matchup_gate: bool = False
    specialist_triggers: tuple = ('giant','prince','dark-prince')
    general_triggers: tuple = ()
    defensive_margin: float = 0.
    attack_margin: float = 0.
    near_tower_margin: float = 0.
    search_width: int = 0
    horizon: int = 125
    reserve_value: float = 155.
    defense_value: float = 1.35
    student_value: float = 0.
    correct_board: bool = True
    diverse: bool = True
    by_cost: bool = True

    def __post_init__(self):
        scores=(self.rollout_threshold,self.defensive_margin,self.attack_margin,self.near_tower_margin,
                self.reserve_value,self.defense_value,self.student_value)
        if not (0 <= self.rollout_weight <= 1 and 5 <= self.rollout_interval <= 60
                and (self.search_width==0 or 2<=self.search_width<=32) and 20 <= self.horizon <= 240
                and all(np.isfinite(s) for s in scores) and self.reserve_value>=0 and self.defense_value>0):
            raise ValueError("Invalid full-match policy budget")

    @classmethod
    def load(cls,path):
        return cls(**json.loads(path.read_text(encoding="utf8"))["parameters"])


def tile_move(card,ready,obs,point,reason):
    slot=ready[card]
    legal=np.asarray(obs.action_mask.placement_masks[str(slot)]['row_major'],dtype=bool)
    ax,ay=point
    col,row=int(ax//1000),int(ay//1000)
    if 0<=col<18 and 0<=row<32 and legal[row,col]:
        return Move(False,slot,card,col*1000+500,row*1000+500,reason)
    if not legal.any():
        return None
    dist=(X-ax)**2+(Y-ay)**2
    row,col=np.unravel_index(np.argmin(np.where(legal,dist,np.inf)),dist.shape)
    return Move(False,slot,card,int(X[row,col]),int(Y[row,col]),reason)


def target_tower(towers,units):
    enemy=[t for t in towers if t.owner==1 and t.active]
    princess=[t for t in enemy if t.tower_kind!='king']
    if not princess:
        return next(iter(enemy),None)
    def cost(t):
        left=t.position[0]<9000
        defenders=sum(1 for u in units if (u.x<9000)==left and u.y>12500 and can_hit(u))
        return t.hitpoints/max(1,t.max_hitpoints)+min(.10,.035*defenders)
    return min(princess,key=cost)
