from dataclasses import dataclass
import math

import numpy as np

from defensive_placement import X, Y, projected
from fit_tile_prior import CARDS


SIEGE_RANGE = {'x-bow': 11500, 'mortar': 11500}
SPLASH = {'valkyrie','wizard','baby-dragon','witch','dark-prince','bowler','executioner',
          'mega-knight','bomber','electro-giant','royal-ghost','fire-spirit','skeleton-king'}
HEAVY_MELEE = {'pekka','mini-pekka','prince','bandit','knight','elite-barbarians','lumberjack'}
FRAGILE_GROUPS = {'skeletons','skeleton-army','goblin-gang','goblins','spear-goblins',
                  'bats','minions','minion-horde','wall-breakers','bush-goblin'}


@dataclass(frozen=True)
class Unit:
    id: int
    card_id: int | None
    name: str
    x: int
    y: int
    air: bool | None
    buildings: bool
    hp: float | None
    vx: float = 0.
    vy: float = 0.
    structure: bool = False
    targetable: bool | None = None
    first_y: int | None = None


def building_name(name):
    compact = name.lower().split(':')[-1].split('_ev')[0].split('-ev')[0].replace('-', '')
    return {'xbow':'x-bow','mortar':'mortar','tesla':'tesla','bushgoblin':'bush-goblin'}.get(compact, name)


def can_hit(unit):
    return unit.targetable is not False and (unit.name != 'tesla' or unit.targetable is True)


def siege_tower(unit, towers):
    reach = SIEGE_RANGE.get(unit.name)
    if reach is None:
        return None
    threatened = [t for t in towers if t.owner == 0 and t.active
                  and math.dist(t.position, (unit.x,unit.y)) <= reach + 1650]
    return min(threatened, key=lambda t: math.dist(t.position,(unit.x,unit.y)), default=None)


def threat_priority(unit, towers):
    if not can_hit(unit):
        return math.inf
    if unit.name in SIEGE_RANGE:
        return 6500 if siege_tower(unit,towers) else math.inf
    if unit.structure or unit.y > (21000 if unit.buildings else 17800):
        return math.inf
    return unit.y - (2800 if unit.buildings else 0)


class ResponseTable:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as model:
            self.names = {str(name):i for i,name in enumerate(model['enemies'])}
            self.card, self.tile = model['card_logp'], model['tile_logp']
            self.support = model['support']
            self.gains = model['sim_gain'] if 'sim_gain' in model else None
            self.offsets = model['sim_offset'] if 'sim_offset' in model else None
            shape = (len(self.names), 3, len(CARDS))
            if (tuple(model['cards']) != CARDS or self.card.shape != shape or self.tile.shape != (*shape,32,18)
                    or self.support.shape != shape[:2] or not np.isfinite(self.tile).all()
                    or not np.isfinite(self.card).all()):
                raise ValueError('Invalid defensive response table')
            if (self.gains is None) != (self.offsets is None):
                raise ValueError('Incomplete simulator responses')
            if self.gains is not None and (
                    self.gains.shape != (len(self.names),len(CARDS)) or
                    self.offsets.shape != (*self.gains.shape,2) or not np.isfinite(self.gains).all()
                    or not np.isfinite(self.offsets).all() or np.abs(self.offsets).max()>2):
                raise ValueError('Invalid simulator responses')

    @staticmethod
    def bridge_weight(unit):
        return min(1., max(0., (unit.y - 11000) / 4500))

    def lookup(self, unit):
        i = self.names.get(unit.name)
        depth = unit.first_y if unit.first_y is not None else unit.y
        bucket = int(depth > 13000) + int(depth > 21000)
        if i is None or self.support[i,bucket] < 8:
            return None, None
        tiles = self.tile[i,bucket]
        scores = self.card[i,bucket]
        if self.gains is not None:
            scores = scores + self.bridge_weight(unit) * self.gains[i]
        return scores, tiles if unit.x < 9000 else tiles[:,:,::-1]

    def adjust_aim(self, aim, unit, card):
        i = self.names.get(unit.name)
        if self.offsets is None or i is None:
            return aim
        dx,dy = self.offsets[i,CARDS.index(card)]
        scale = 1000*self.bridge_weight(unit)
        return aim[0]+float(dx)*scale*(1 if unit.x<9000 else -1), aim[1]+float(dy)*scale

    def contextual(self, unit, head, events, hand, tick):
        scores, tiles = self.lookup(unit)
        if head is None:
            return scores, tiles, None
        logits, hidden = head.predict(unit,events,hand,tick)
        i = self.names.get(unit.name)
        if scores is None:
            return logits, tiles, hidden
        depth = unit.first_y if unit.first_y is not None else unit.y
        bucket = int(depth>13000) + int(depth>21000)
        return scores + .7 * (logits - self.card[i,bucket]), tiles, hidden


def fireball_target(ready_rows, focus, enemies, towers):
    """Require a damaging hit on the focus, then maximize actual targets inside the blast."""
    if not can_hit(focus):
        return None, 0.
    fx, fy = projected(focus, 18)
    radius_sq = 2500**2
    valid = np.asarray(ready_rows, dtype=bool) & ((X-fx)**2 + (Y-fy)**2 <= radius_sq)
    if not valid.any():
        return None, 0.
    hits = np.zeros(X.shape)
    siege_hits = np.zeros(X.shape, dtype=bool)
    for unit in enemies:
        if not can_hit(unit):
            continue
        x,y = projected(unit,18)
        if abs(x-fx)>5000 or abs(y-fy)>5000:
            continue
        siege = siege_tower(unit,towers) is not None
        weight = 3.5 if siege else (1.4 if unit.structure else 1.)
        if unit.name == 'balloon' and y < 13500:
            weight = 2.
        if unit.name in {'skeletons','bats','skeleton-army'}:
            weight = .25
        covered = (X-x)**2+(Y-y)**2 <= radius_sq
        hits += weight * covered
        if siege:
            siege_hits |= covered
    for tower in towers:
        if tower.owner == 1 and tower.active:
            x,y = tower.position
            hits += .4 * ((X-x)**2+(Y-y)**2 <= radius_sq)
    if (valid & siege_hits).any():
        valid &= siege_hits
    score = hits - .05 * ((X-fx)**2+(Y-fy)**2) / radius_sq
    row,col = np.unravel_index(np.argmax(np.where(valid,score,-np.inf)), score.shape)
    return (int(X[row,col]),int(Y[row,col])), float(hits[row,col])


def choose_counter(ready, unit, group, card_prior, spell_value):
    siege = unit.name in SIEGE_RANGE
    cheap_group = sum(u.air is False and ((u.hp or 99999) <= 420 or u.name=='bush-goblin') for u in group)
    scores = {}
    for card in ready:
        score = 0.
        if card == 'fireball':
            if spell_value < 1.9:
                continue
            score = (1.1 if siege else .5) + min(4.5, spell_value) * .72
        elif card == 'the-log':
            if cheap_group < 2 or unit.air is not False or unit.structure:
                continue
            score = 2.3 + min(4,cheap_group)*.38
        elif card == 'hog-rider':
            if not siege:
                continue
            score = 2.35
        elif card == 'ice-golem':
            if unit.buildings or ((unit.air is True or unit.structure) and not siege):
                continue
            score = 1.75 if siege or unit.name in SPLASH else .1
        elif card == 'cannon':
            if (unit.air is not False and not unit.buildings) or siege:
                continue
            score = (1.9 if unit.air else 3.5) if unit.buildings else .65
        elif card == 'musketeer':
            score = 3.4 if unit.air is True else (1.8 if siege or unit.name in SPLASH else .8)
        elif card == 'skeletons':
            if unit.air is not False or siege or unit.name in SPLASH:
                continue
            if unit.buildings and (unit.name!='giant' or any(member.name in SPLASH for member in group)):
                continue
            score = 2.3 if unit.name in HEAVY_MELEE else .2
        elif card == 'ice-spirit':
            if siege or unit.name=='electro-giant' or (unit.y>14500 and unit.name not in FRAGILE_GROUPS):
                continue
            score = 2.2 if unit.name in FRAGILE_GROUPS else .15
        else:
            continue
        scores[card] = score + (float(card_prior[CARDS.index(card)]) if card_prior is not None else 0.)
    return max(scores,key=scores.get,default=None)
