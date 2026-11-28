import math

import numpy as np


ROWS, COLS = np.indices((32, 18))
X = COLS * 1000 + 500
Y = ROWS * 1000 + 500
PRIOR_INDEX = ((Y * 18 // 32000) * 32 + X * 32 // 18000)


def projected(unit, ticks):
    dx, dy = unit.vx * ticks, unit.vy * ticks
    scale = min(1., 2200 / max(1., math.hypot(dx, dy)))
    return (min(17999, max(0, unit.x + dx * scale)),
            min(31999, max(0, unit.y + dy * scale)))


def aim_for(card, unit, group, towers):
    side = 'left' if unit.x < 9000 else 'right'
    own = {tower.tower_kind: tower for tower in towers if tower.owner == 0 and tower.active}
    tower = own.get('princess_' + side) or own.get('king')
    tx, ty = tower.position if tower else (9000, 3000)
    x, y = projected(unit, 18)
    inward = 1 if unit.x < 9000 else -1
    if card == 'fireball':
        points = [projected(member, 18) for member in group]
        return tuple(np.mean(points, axis=0))
    if card == 'the-log':
        return x, y - 1500
    if card == 'cannon':
        return (min(10500, max(7500, x + 3000 * inward)),
                min(12500, max(ty + 2200, y - 1800)))
    if card == 'musketeer':
        return x + 2400 * inward, max(2000, y - 4100)
    dx, dy = tx - x, ty - y
    length = max(1., math.hypot(dx, dy))
    intercept = 1150 if card == 'skeletons' else 800
    return x + intercept * dx / length, y + intercept * dy / length


def closest_defensive_tile(card, rows, aim, prior, towers):
    legal = np.array(rows, dtype=bool, copy=True)
    if card not in {'the-log', 'fireball'}:
        legal &= Y < 16000
        for tower in towers:
            if tower.owner == 0 and tower.active:
                tx, ty = tower.position
                half_width = 2000 if tower.tower_kind == 'king' else 1500
                legal &= (abs(X - tx) >= half_width) | (abs(Y - ty) >= half_width)
    if not legal.any():
        return None
    distance = ((X - aim[0]) / 1000)**2 + ((Y - aim[1]) / 1000)**2
    # Bound the replay preference so a popular tile cannot pull a defender away from its intercept.
    score = -distance + 0.2 * np.tanh(prior[PRIOR_INDEX] / 4)
    row, col = np.unravel_index(np.argmax(np.where(legal, score, -np.inf)), score.shape)
    return int(X[row, col]), int(Y[row, col])
