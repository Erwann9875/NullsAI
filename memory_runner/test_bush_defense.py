from types import SimpleNamespace as NS

import numpy as np

from bush_defense import BushResponse, CARDS, SHAPE
from defense_responses import Unit, building_name


def test_bush_head_uses_an_affordable_legal_alternative_to_cannon(tmp_path):
    table=np.zeros(SHAPE,np.float32)
    table[...,CARDS.index('cannon'),:]=3
    table[...,CARDS.index('the-log'),:]=2
    table[...,CARDS.index('skeletons'),:]=1
    path=tmp_path/'bush.npz'
    np.savez(path,scores=table,cards=CARDS)
    policy=BushResponse(path)
    valid=np.ones((32,18),bool)
    obs=NS(towers=[NS(owner=0,active=True,tower_kind='princess_right',hitpoints=785,max_hitpoints=4858)],
           action_mask=NS(placement_masks={'1':{'row_major':valid},'2':{'row_major':valid}}))
    group=[Unit(41,26000097,'bush-goblin',13769,9558,False,False,539)]
    assert policy.choose(obs,group,{'skeletons':1,'the-log':2}).card=='the-log'
    obs.action_mask.placement_masks['2']={'row_major':np.zeros_like(valid)}
    assert policy.choose(obs,group,{'skeletons':1,'the-log':2}).card=='skeletons'
    assert policy.choose(obs,group,{}).wait
    assert building_name('BushGoblin')=='bush-goblin'
