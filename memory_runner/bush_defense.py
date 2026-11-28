"""Small action-value table for goblins visibly revealed by Suspicious Bush."""
import numpy as np


BUSH_GOBLIN_ID = 2104309777
CARDS = ('skeletons', 'ice-spirit', 'the-log', 'ice-golem', 'musketeer', 'fireball', 'cannon')
COST = (1, 1, 2, 2, 4, 4, 3)
SHAPE = (2, 2, 2, len(CARDS), 2)


def context(y, count, tower_fraction):
    return int(y > 11000), int(count > 1), int(tower_fraction < .45)


def positions(card, x, y, left):
    inward = 1 if left else -1
    tower_x = 3500 if left else 14500
    if card == 'the-log':
        points = [(x, y-3700), (x, y-2400)]
    elif card == 'fireball':
        points = [(x, y-1700), (x, y-800)]
    elif card in ('cannon', 'musketeer'):
        points = [(tower_x+3000*inward, 9500), (x+1800*inward, y-2500)]
    else:
        points = [(x+inward*900, y-1100), (x+inward*2600, y-1700)]
    result = []
    for a,b in points:
        px = int(max(500,min(17500,a))//1000)*1000+500
        py = int(max(500,min(15500,b))//1000)*1000+500
        if card not in ('fireball','the-log') and abs(px-tower_x)<1500 and abs(py-6500)<1500:
            px = tower_x+2000*inward
        result.append((px,py))
    return result


class BushResponse:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.scores = data['scores']
            if (tuple(data['cards']) != CARDS or self.scores.shape != SHAPE
                    or not np.isfinite(self.scores).all()):
                raise ValueError('Invalid Bush Goblin defense table')

    def choose(self, obs, group, ready):
        from live_decision import Move
        from match_strategy import tile_move
        lead = min(group, key=lambda u:u.y)
        left = lead.x < 9000
        kind = 'princess_left' if left else 'princess_right'
        tower = next((t for t in obs.towers if t.owner==0 and t.tower_kind==kind and t.active), None)
        if tower is None:
            return None
        x = sum(u.x for u in group)/len(group)
        scores = self.scores[context(lead.y,len(group),tower.hitpoints/tower.max_hitpoints)]
        best = Move(reason='reserve_against_bush_goblins')
        best_score = 0.
        for i, card in enumerate(CARDS):
            if card not in ready:
                continue
            for j, point in enumerate(positions(card,x,lead.y,left)):
                score = scores[i,j]
                if score <= best_score:
                    continue
                candidate = tile_move(card,ready,obs,point,'defend_bush_goblins')
                if candidate and abs(candidate.x-point[0])+abs(candidate.y-point[1]) <= 1500:
                    best, best_score = candidate, score
        return best
