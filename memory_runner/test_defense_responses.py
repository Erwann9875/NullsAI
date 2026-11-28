from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from defense_responses import Unit, choose_counter, fireball_target, threat_priority
from memory_policy import MemoryPolicy


TOWERS = [SimpleNamespace(owner=0,active=True,tower_kind='princess_left',position=(3500,6500))]
ROWS = np.ones((32,18),dtype=bool)


def unit(name,x,y,**changes):
    return replace(Unit(1,None,name,x,y,False,False,1000),**changes)


@pytest.mark.parametrize('state',[False,None])
def test_underground_or_unconfirmed_tesla_never_justifies_fireball(state):
    tesla=unit('tesla',4500,17500,structure=True,targetable=state)
    assert fireball_target(ROWS,tesla,[tesla],TOWERS)==(None,0.)
    goblin=unit('goblins',4500,16000)
    _,value=fireball_target(ROWS,goblin,[goblin,tesla],TOWERS)
    assert value==1.
    assert choose_counter(['fireball'],goblin,[goblin],None,value) is None


def test_cluster_spell_covers_attacking_xbow_with_its_support():
    bow=unit('x-bow',3500,18500,structure=True)
    support=unit('archers',6500,16500)
    tesla=unit('tesla',10500,17500,structure=True)
    allowed=ROWS.copy()
    allowed[:,5]=False
    point,value=fireball_target(allowed,bow,[bow,support,tesla],TOWERS)
    assert allowed[point[1]//1000,point[0]//1000]
    assert all(np.hypot(point[0]-u.x,point[1]-u.y)<=2500 for u in (bow,support))
    assert value==4.5
    assert choose_counter(['hog-rider','fireball'],bow,[bow],None,value)=='fireball'
    alternative,_=fireball_target(allowed,support,[support,bow,tesla],TOWERS)
    assert np.hypot(alternative[0]-bow.x,alternative[1]-bow.y)<=2500
    assert np.isfinite(threat_priority(bow,TOWERS))
    assert not np.isfinite(threat_priority(replace(bow,y=27000),TOWERS))


def test_counters_change_with_target():
    hand=['skeletons','musketeer','cannon','hog-rider']
    for target,card in [(unit('mini-pekka',3500,14500),'skeletons'),
                        (unit('hog-rider',3500,14500,buildings=True),'cannon'),
                        (unit('balloon',3500,14500,air=True,buildings=True),'musketeer')]:
        assert choose_counter(hand,target,[target],None,0.)==card


def test_skeletons_can_damage_an_unsupported_giant():
    giant=unit('giant',3500,14000,buildings=True)
    assert choose_counter(['skeletons'],giant,[giant],None,0.)=='skeletons'
    assert choose_counter(['skeletons'],giant,[giant,unit('wizard',4000,14500)],None,0.) is None


def test_reader_delivers_visible_siege_to_planner_without_loading_checkpoint():
    policy=object.__new__(MemoryPolicy)
    policy.metadata={(35,27):('x-bow',False,False)}
    policy.inputs=SimpleNamespace(enemy_elixir=5)
    policy.builder=SimpleNamespace(pending=[])
    policy.first_y={}
    building=SimpleNamespace(visible=True,owner=1,entity_kind='building',hitpoints=1500,
                             native_data_global_id=35,card_id=27,position=(3500,18500),velocity=None,entity_id=90)
    obs=SimpleNamespace(towers=[],entities=[building],tick=78,players=[SimpleNamespace(elixir_exact=5.)])
    units,features=policy.read_units(obs)
    assert len(units)==1 and units[0].structure and features.shape==(58,)
    assert np.isfinite(threat_priority(units[0],TOWERS))
    building.visible=False
    assert not policy.read_units(obs)[0]


def test_visible_buildings_in_public_battle_frame_keep_their_card_and_tile():
    from native_runner.live_probe.live_observation import LiveObservationBuilder
    from native_runner.tests.live_fixtures import battle_payload

    builder=LiveObservationBuilder(allow_unknown_entities=True)
    raw=battle_payload()
    for idx,(card,archetype,x) in enumerate(((27000008,35000010,3500),(27000006,35000008,5500))):
        raw['objects'].append(dict(nativeObjectId=100+idx,owner=1,cardId=card,dataGlobalId=archetype,
                                   x=x,y=18500,targetX=x,targetY=18500,hp=1000,maxHp=1000))
    raw['returned']=len(raw['objects'])
    obs=builder.build(raw)
    policy=object.__new__(MemoryPolicy)
    policy.inputs=SimpleNamespace(enemy_elixir=5.,card=lambda cid:builder.bundle.card_specs[cid].name)
    policy.builder=builder
    policy.metadata={}
    policy.first_y={}
    units,_=policy.read_units(obs)
    assert [(u.name,u.x,u.structure) for u in units]==[('x-bow',3500,True),('tesla',5500,True)]
    assert np.isfinite(threat_priority(units[0],obs.towers))
    assert np.isinf(threat_priority(units[1],obs.towers))
