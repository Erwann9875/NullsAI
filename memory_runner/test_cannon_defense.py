from types import SimpleNamespace as NS

import numpy as np
import pytest

from defense_responses import Unit
from memory_policy import cannon_defense


def unit(name, x=3262, y=16500, hp=6000, air=False):
    return Unit(1,None,name,x,y,air,False,hp)


def board():
    mask=np.zeros((32,18),bool)
    mask[:16]=True
    return NS(towers=[],action_mask=NS(placement_masks={'0':{'row_major':mask}}))


def test_underfunded_push_holds_log_and_cyclers_until_cannon_is_affordable():
    push=[unit('mega-knight'),unit('firecracker',y=17500,hp=486)]
    hand=['cannon','the-log','ice-spirit','skeletons']
    response,protect=cannon_defense(board(),push,[],hand,{'the-log':1,'ice-spirit':2,'skeletons':3},2.03)
    assert protect and response[0].wait and response[1] is push[0]
    assert response[0].reason=='reserve_cannon_for_push'


@pytest.mark.parametrize('enemy_x,expected_col',[(3262,7),(14738,10)])
def test_cannon_pulls_towards_center_from_either_lane(enemy_x,expected_col):
    threat=unit('mega-knight',x=enemy_x)
    (move,focus),_=cannon_defense(board(),[threat],[],['cannon'],{'cannon':0},3)
    assert focus is threat and (move.card,move.x//1000,move.y//1000)==('cannon',expected_col,8)


def test_reserve_prevents_unfunded_hog_without_spending_cannon_on_lumberjack():
    threat=unit('lumberjack',x=13649,y=15676,hp=2049)
    args=(board(),[threat],[],['cannon','hog-rider'],{'cannon':0,'hog-rider':1})
    assert cannon_defense(*args,4.65)==(None,True)
    assert cannon_defense(*args,7)==(None,False)


@pytest.mark.parametrize('balance,expected',[(4.65,''),(7.,'hog-rider')])
def test_live_planner_applies_reserve_to_a_rollout_that_prefers_hog(balance,expected):
    from defense_trials import TOWERS
    from live_decision import DECK
    from match_strategy import MatchStrategy
    from memory_policy import MemoryPolicy
    hand=['cannon','hog-rider','the-log','ice-golem']
    policy=object.__new__(MemoryPolicy)
    policy.inputs=NS(card=lambda c:c,events=[])
    policy.builder=NS(pending=[])
    policy.elixir_cap=None
    policy.allies=[]
    policy.match_strategy=MatchStrategy(matchup_gate=False,rollout_weight=1,search_width=0)
    policy.forecast=None
    policy.next_rollout_tick=0
    policy.rollout_head=NS(scores=lambda rows: 5*rows[:,1+DECK.index('hog-rider')]-1)
    towers=[NS(**vars(t),hitpoints=3000,max_hitpoints=3000) for t in TOWERS]
    placement=board().action_mask.placement_masks['0']
    obs=NS(tick=263,towers=towers,players=[NS(hand=hand,elixir_exact=balance)],
           action_mask=NS(reasons={'effective_elixir':balance},hand_slots=[True]*4,
                          placement_masks={str(i):placement for i in range(4)}))
    move,_=policy.plan(obs,[unit('lumberjack',x=13649,y=15676,hp=2049)],None)
    assert move.card==expected


def test_absent_card_existing_cover_and_air_leave_the_other_defenses_available():
    threat=unit('mega-knight')
    friendly=unit('cannon',x=7500,y=8500,hp=900)
    assert cannon_defense(board(),[threat],[friendly],['cannon'],{'cannon':0},5)==(None,False)
    assert cannon_defense(board(),[threat],[],['musketeer'],{'musketeer':0},5)==(None,False)
    assert cannon_defense(board(),[unit('balloon',air=True)],[],['cannon'],{'cannon':0},5)==(None,False)
    obs=board()
    obs.action_mask.placement_masks['0']['row_major'][:]=False
    assert cannon_defense(obs,[threat],[],['cannon'],{'cannon':0},5)[0] is None
