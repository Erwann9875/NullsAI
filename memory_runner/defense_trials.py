"""Run paired, single-push counterfactuals in CRForge using public defender observations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from defense_responses import CARDS, ResponseTable, Unit, choose_counter, fireball_target, threat_priority
from defensive_placement import aim_for, closest_defensive_tile
from response_context import ContextHead


THREATS = ('x-bow','mortar','hog-rider','balloon','giant','royal-giant','electro-giant',
           'battle-ram','mini-pekka','knight','prince','valkyrie','baby-dragon','minion-horde',
           'skeleton-army','archers','wizard','goblin-gang')
OFFSETS = ((0,0),(0,-2),(0,2),(-2,0),(2,0))
TOWERS = tuple(SimpleNamespace(owner=owner, tower_kind=kind, active=True, position=(x,y if not owner else 32000-y))
               for owner in (0,1) for kind,x,y in [('king',9000,3000),('princess_left',3500,6500),
                                                    ('princess_right',14500,6500)])


def tick(arena):
    return arena.schema.field(arena.observations[0,0],'tick')


def hand(arena, side):
    ids = arena.schema.field(arena.observations[0,side],'hand_card')
    return [arena.vocabulary[i] if i>=0 else '' for i in ids]


def nearest(mask, col, row):
    y,x = np.nonzero(mask)
    i = np.argmin((x-col)**2+(y-row)**2)
    return int(x[i]),int(y[i])


def prepare(arena, seed, threat, native, lookup):
    rng = random.Random(seed)
    own = [native[c] for c in CARDS]
    filler = [native[c] for c in ('knight','archers','hog-rider','fireball','ice-spirit','skeletons','cannon','the-log')]
    enemy = [native[threat]] + [c for c in filler if c != native[threat]][:7]
    rng.shuffle(own)
    rng.shuffle(enemy)
    def reset():
        arena.reset([seed],own,enemy)
        arena.actions.fill(0)
        while tick(arena) < 180:
            arena.step()
    reset()
    current = hand(arena,1)
    if native[threat] not in current:
        a,b = enemy.index(native[threat]),enemy.index(current[0])
        enemy[a],enemy[b] = enemy[b],enemy[a]
        reset()
    slot = hand(arena,1).index(native[threat])
    masks = arena.masks[0,1,1:2305].reshape(4,32,18)
    while not masks[slot].any():
        arena.step()
        if tick(arena)>700:
            raise RuntimeError(f'Could not deploy {threat}')
    col = rng.choice((3,4,5,12,13,14))
    row = rng.choice((13,14,15))
    x,y = nearest(masks[slot],col,row)
    arena.actions[0,1] = arena.schema.play(slot,x,y)
    arena.step()
    arena.actions.fill(0)
    last = {}
    first = {}
    s = arena.schema
    for _ in range(90):
        obs = arena.observations[0,0]
        units = []
        for obj in s.entities(obs):
            def val(key):
                return int(obj[s.entity[key]])
            if val('kind')!=s.kinds['character'] or val('side')!=1 or val('hp')<=0:
                continue
            flags = val('flags')
            if flags & (s.flags['hidden']|s.flags['underground']|s.flags['invisible']):
                continue
            ident,x,y = val('id'),val('x'),val('y')
            first.setdefault(ident,y)
            prev = last.get(ident,(x,y,tick(arena)-1))
            dt = max(1,tick(arena)-prev[2])
            facts = lookup[threat]
            u = Unit(ident,None,threat,x,y,bool(flags&s.flags['air']),facts['target']=='buildings',
                     val('hp'),(x-prev[0])/dt,(y-prev[1])/dt,bool(flags&s.flags['building']),True,first[ident])
            last[ident]=(x,y,tick(arena))
            units.append(u)
        if units:
            focus=min(units,key=lambda u:threat_priority(u,TOWERS))
            if np.isfinite(threat_priority(focus,TOWERS)):
                return focus,units
        arena.step()
    raise RuntimeError(f'No public threatening {threat} at {col},{row}')


def options(arena, focus, units, native, response, prior, head):
    masks = arena.masks[0,0,1:2305].reshape(4,32,18)
    own = hand(arena,0)
    ready = {card:own.index(name) for card,name in native.items() if card in CARDS and name in own and masks[own.index(name)].any()}
    groups = [u for u in units if (u.x-focus.x)**2+(u.y-focus.y)**2<=2700**2]
    initial=focus.first_y if focus.first_y is not None else focus.y
    events=[dict(side='opponent',seconds=(tick(arena)-max(0,initial-focus.y)//55)/20,
                 card=focus.name,x=focus.x,y=initial)]
    all_cards=[c for c in CARDS if native[c] in own]
    card_logs,tiles,hidden=response.contextual(focus,head,events,all_cards,tick(arena))
    spell,worth=(None,0.)
    if 'fireball' in ready:
        spell,worth=fireball_target(masks[ready['fireball']],focus,units,TOWERS)
    entries = [('wait',0,0)]
    for card,slot in ready.items():
        if choose_counter([card],focus,groups,card_logs,worth) is None:
            continue
        idx=CARDS.index(card)
        learned=tiles[idx] if tiles is not None else None
        if hidden is not None:
            surface=head.tiles(hidden,card,focus.x>=9000)
            learned=surface if learned is None else .5*(learned+surface)
        for variant,(dx,dy) in enumerate(OFFSETS):
            if card in ('fireball','the-log') and variant:
                continue
            if card=='fireball':
                position=spell
            else:
                ax,ay=aim_for(card,focus,groups,TOWERS)
                aim=(ax + dx*1000*(1 if focus.x<9000 else -1), ay + dy*1000)
                context=1+int(focus.x>=9000) if focus.buildings and focus.air is False else 0
                position=closest_defensive_tile(card,masks[slot],aim,prior[idx,context],TOWERS,
                                                learned if card!='the-log' else None)
            if position is None:
                continue
            x,y=position
            entries.append((card,variant,arena.schema.play(slot,x//1000,y//1000)))
    pick=choose_counter(ready,focus,groups,card_logs,worth) or 'wait'
    return entries,pick,dict(card_scores=None if card_logs is None else card_logs.tolist(),spell_value=worth)


def baseline_policy(path, firstlight, reference, prior):
    from runtime import configure
    configure(firstlight,reference)
    spec=importlib.util.spec_from_file_location('comparison_policy',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    actor=object.__new__(module.MemoryPolicy)
    actor.prior=prior
    actor.elixir_cap=actor.recent=None
    actor.inputs=SimpleNamespace(card=lambda c:c,events=[],seen=[],enemy_elixir=5.)
    actor.builder=SimpleNamespace(pending=[])
    from live_decision import tile_tables,Move
    actor.points,actor.tile_masks=tile_tables()
    actor.cols,actor.rows=(actor.points//1000).T
    # This benchmark isolates one defensive response. Pass when the previous runner would attack.
    actor.fast=SimpleNamespace(choose=lambda *a,**kw:Move())
    return actor


def baseline_action(actor,arena,units,native):
    lookup={v:k for k,v in native.items() if k in CARDS}
    cards=[lookup.get(name,'') for name in hand(arena,0)]
    view=arena.observations[0,0]
    field=lambda key:arena.schema.field(view,key)
    elixir=(field('elixir')-field('elixir_reserved'))/10000
    masks=arena.masks[0,0,1:2305].reshape(4,32,18)
    own=SimpleNamespace(hand=cards,elixir_exact=elixir)
    mask=SimpleNamespace(hand_slots=masks.any(axis=(1,2)),reasons={'effective_elixir':elixir},
                         placement_masks={str(i):{'row_major':m} for i,m in enumerate(masks)})
    obs=SimpleNamespace(players=[own],action_mask=mask,towers=TOWERS,tick=tick(arena))
    move,_=actor.plan(obs,[u for u in units if not u.structure],np.zeros(58))
    return (('wait',0) if move.wait else (move.card,arena.schema.play(move.slot,move.x//1000,move.y//1000)))


def rollout(source, forks, entries):
    unique=list(dict.fromkeys(action for _,_,action in entries))
    scores={}
    for start in range(0,len(unique),forks.count):
        actions=unique[start:start+forks.count]
        forks.fork_from(source,0,-1,list(range(forks.count)))
        forks.actions.fill(0)
        forks.actions[:len(actions),0]=actions
        forks.step()
        forks.actions.fill(0)
        for _ in range(80):
            forks.step()
        for index,action in enumerate(actions):
            info=forks.info(index)
            seat=forks.observations[index,0]
            if info['fault'] or info['illegal_0'] or info['illegal_1']:
                raise RuntimeError(f'Invalid simulator branch {info}')
            def field(name):
                return max(0,int(forks.schema.field(seat,name)))
            hp=sum(field('own_'+name+'_hp') for name in ('left','right','king'))
            princess=sum(field('own_'+name+'_hp') for name in ('left','right'))
            lost=sum(field('own_'+name+'_hp')==0 for name in ('left','right'))
            enemy=sum(field('enemy_'+name+'_hp') for name in ('left','right','king'))
            scores[action]=dict(hp=hp,princess=princess,lost=lost,enemy_hp=enemy)
    return [dict(card=c,offset=o,action=a,**scores[a]) for c,o,a in entries]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('reference','simulator','tables','responses','tile-prior','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--seeds',type=int,required=True)
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--baseline-source',type=Path)
    p.add_argument('--firstlight',type=Path)
    p.add_argument('--context',type=Path)
    p.add_argument('--threats',nargs='+',default=THREATS,choices=THREATS)
    a=p.parse_args()
    from support import limit_cpu
    limit_cpu()
    java=shutil.which('java')
    if java:
        os.environ.setdefault('JAVA_HOME',str(Path(java).resolve().parents[1]))
    sys.path.insert(0,str(a.simulator.resolve()/'python'))
    from crforge_env import VectorBattle, LadderConfig
    catalog=json.loads((a.reference/'engine/data/cards.json').read_text(encoding='utf8'))
    lookup={r['id'].replace('_','-'):r for r in catalog}
    native={k:v['sc_name'] for k,v in lookup.items()}
    response=ResponseTable(a.responses)
    head=ContextHead(a.context) if a.context else None
    with np.load(a.tile_prior) as data:
        prior=data['log_probs'].reshape(8,3,576)
    baseline=baseline_policy(a.baseline_source,a.firstlight,a.reference,prior) if a.baseline_source else None
    config=dict(install=a.simulator/'env/build/install/env',tables=a.tables,
                config=LadderConfig(decision_ticks=10,level=11,tower_level=11,mask_version=2),
                jvm_args=('-Xmx256m','-XX:ActiveProcessorCount=1','-XX:+UseSerialGC'))
    with a.out.open('x',encoding='utf8') as stream, VectorBattle(1,1,**config) as source, VectorBattle(8,1,**config) as forks:
        forks.set_autoreset(False)
        for i in range(a.seeds):
            for j,threat in enumerate(a.threats):
                seed=a.start+100*i+j
                focus,units=prepare(source,seed,threat,native,lookup)
                entries,pick,decision=options(source,focus,units,native,response,prior,head)
                original=None
                if baseline is not None:
                    original,action=baseline_action(baseline,source,units,native)
                    entries.append(('baseline',0,action))
                results=rollout(source,forks,entries)
                stream.write(json.dumps(dict(seed=seed,enemy=threat,focus=vars(focus),group=[vars(u) for u in units],
                                             hand=hand(source,0),baseline_card=original,policy_pick=pick,
                                             decision=decision,results=results))+'\n')
                stream.flush()
            print(f'Completed {i+1}/{a.seeds} seed groups',flush=True)


if __name__=='__main__':
    main()
