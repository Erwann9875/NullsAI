from types import SimpleNamespace

import numpy as np

from defensive_placement import aim_for, closest_defensive_tile, projected


def unit(x, y, vx=0, vy=0):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy)


def test_all_native_rows_are_reachable_when_legal():
    prior = np.zeros(576)
    for row in range(32):
        legal = np.zeros((32, 18), dtype=bool)
        legal[row, 3] = True
        center = (3500, row * 1000 + 500)
        assert closest_defensive_tile('fireball', legal, center, prior, []) == center


def test_prior_cannot_displace_intercept_or_select_illegal_tile():
    legal = np.ones((32, 18), dtype=bool)
    prior = np.zeros(576)
    prior[0] = 30
    assert closest_defensive_tile('skeletons', legal, (3500, 10500), prior, []) == (3500, 10500)
    legal[10, 3] = False
    assert closest_defensive_tile('skeletons', legal, (3500, 10500), prior, []) != (3500, 10500)
    assert closest_defensive_tile('skeletons', legal & False, (3500, 10500), prior, []) is None


def test_spell_leads_the_moving_group_and_prediction_is_bounded():
    group = [unit(3000, 14000, 0, -50), unit(4000, 13000, 0, -50)]
    assert aim_for('fireball', group[0], group, []) == (3500, 12600)
    assert projected(unit(8000, 12000, 0, -1000), 18) == (8000, 9800)


def test_melee_uses_surviving_king_after_lane_tower_is_lost():
    defender = unit(4500, 7500)
    king = SimpleNamespace(owner=0, active=True, tower_kind='king', position=(9000, 3000))
    left = SimpleNamespace(owner=0, active=True, tower_kind='princess_left', position=(3500, 6500))
    x_live, _ = aim_for('skeletons', defender, [defender], [king, left])
    left.active = False
    x_lost, y_lost = aim_for('skeletons', defender, [defender], [king, left])
    assert x_live < defender.x < x_lost and y_lost < defender.y
