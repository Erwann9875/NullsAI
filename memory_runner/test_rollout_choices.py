from types import SimpleNamespace as NS

import numpy as np

from match_strategy import target_tower, tile_move
from rollout_features import canonical
from rollout_choices import choices
from run import quick_lookup_seconds


def test_hog_keeps_pressuring_the_weaker_remaining_tower():
    def tower(x,hp):
        return NS(owner=1,active=hp>0,tower_kind='princess_left' if x<9000 else 'princess_right',
                  position=(x,25500),hitpoints=hp,max_hitpoints=3052)
    weak,strong=tower(3500,630),tower(14500,2735)
    defender=NS(x=3500,y=19000,name='prince',targetable=True)
    assert target_tower([strong,weak],[defender]) is weak
    weak.active=False
    assert target_tower([weak,strong],[defender]) is strong


def test_forecast_candidates_snap_to_an_open_tile_in_the_selected_slot():
    mask=np.zeros((32,18),bool)
    mask[14,3]=True
    obs=NS(action_mask=NS(placement_masks={'2':{'row_major':mask}}))
    move=tile_move('hog-rider',{'hog-rider':2},obs,(3500,15500),'test')
    assert (move.slot,move.x,move.y)==(2,3500,14500)
    assert not mask[15,3]


def test_reflection_rotates_tower_hp_and_threats_together():
    rows=np.zeros((1,64),np.float32)
    rows[0,[10,11,13,14,20,26,35]]=[.1,.7,.4,.8,.75,.2,.9]
    normalized=canonical(rows)
    np.testing.assert_allclose(normalized[0,[10,11,13,14,20,26,35]],[.7,.1,.8,.4,.25,.9,.2])
    assert rows[0,20]==.75


def test_after_one_crown_hog_can_reach_the_surviving_princess_or_attack_the_king():
    from defense_trials import TOWERS
    towers=[NS(**{**vars(t),'hitpoints':3000,'max_hitpoints':3000}) for t in TOWERS]
    lost=next(t for t in towers if t.owner==1 and t.tower_kind=='princess_left')
    lost.hitpoints,lost.active=0,False
    legal=np.zeros((32,18),bool)
    legal[:16]=True
    legal[16:22,:9]=True
    obs=NS(towers=towers,tick=2000,action_mask=NS(placement_masks={'0':{'row_major':legal}}))
    moves,_=choices(obs,[],[],{'hog-rider':0},5.)
    tiles={(m.x//1000,m.y//1000) for m in moves if not m.wait}
    assert (14,15) in tiles and (6,21) in tiles


def test_known_kickoff_reaches_full_lookup_without_waiting_out_the_lobby_deadline():
    assert quick_lookup_seconds(120.,True)==20.
    assert quick_lookup_seconds(120.,True,indexed=True)==120.
    assert quick_lookup_seconds(120.,False)==120.
    assert quick_lookup_seconds(0.,True)==0.
