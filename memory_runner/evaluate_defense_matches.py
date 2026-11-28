"""Paired CRForge matches running the two local memory policies' normal decision methods."""
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from types import SimpleNamespace as NS

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

from defense_responses import Unit, CARDS, building_name
from defense_trials import TOWERS
from runtime import configure
from support import limit_cpu

opponents={
    'siege':['x-bow','tesla','archers','knight','skeletons','ice-spirit','fireball','the-log'],
    'balloon':['balloon','baby-dragon','lumberjack','bowler','ice-golem','freeze','barbarian-barrel','tornado'],
    'beatdown':['giant','prince','dark-prince','mega-minion','musketeer','fireball','zap','minions'],
    'hog':list(CARDS)}


def load_previous(path):
    spec=importlib.util.spec_from_file_location('bench_legacy',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod.MemoryPolicy


def actor(cls, paths):
    from native_runner.live_probe.live_observation import LiveObservationBuilder
    return cls(LiveObservationBuilder(allow_unknown_entities=True), paths, controller=object())


class Player:
    def __init__(self,arena,i,seat,template,new,facts,reverse):
        from live_decision import GreedyPolicy
        self.facts,self.reverse=facts,reverse
        self.p=copy.copy(template)
        if getattr(self.p,'forecast',None):
            self.p.forecast=copy.copy(template.forecast)
            self.p.forecast.reset()
        self.p.builder=NS(pending=[])
        self.p.fast=GreedyPolicy(template.fast.agent)
        self.p.inputs=NS(card=lambda c:c,events=[],seen=[],enemy_elixir=5.)
        self.p.recent=None
        self.p.submitted_focus={}
        self.p.elixir_cap=None
        self.arena,self.i,self.seat,self.new=arena,i,seat,new
        self.last_tick=-1
        self.last_decision=-20
        self.events={}
        self.pending=[]
        self.seen={}
        self.origins={}
        self.names={}
        self.timings=[]
        self.plays=Counter()
        self.uses=Counter()
        self.trace=None
        self.teacher=None

    def name(self,idx):
        if idx not in self.names:
            s=self.arena.vocabulary[idx] if idx>=0 else ''
            self.names[idx]=self.reverse.get(s,building_name(s))
        return self.names[idx]

    def action(self):
        s=self.arena.schema
        v=self.arena.observations[self.i,self.seat]
        field=lambda k:s.field(v,k)
        t=int(field('tick'))
        self.p.inputs.enemy_elixir=min(10.,self.p.inputs.enemy_elixir+max(0,t-self.last_tick)*.05/2.8*(2 if t>=2400 else 1))
        self.last_tick=t
        for j in range(min(field('enemy_play_count'),s.enemy_play_rows)):
            idx=int(field('enemy_play_card')[j])
            age=int(field('enemy_play_ticks')[j])
            x,y=int(field('enemy_play_x')[j]),int(field('enemy_play_y')[j])
            if idx<0 or age<=0 or x<0 or y<0:
                continue
            name=self.name(idx)
            key=('opp',t-age,idx,x,y)
            if key not in self.events:
                self.events[key]=dict(side='opponent',event_type='card_play',seconds=(t-age)/20,card=name,x=x,y=y)
                self.p.inputs.enemy_elixir=max(0.,self.p.inputs.enemy_elixir-self.facts.get(name,{}).get('elixir',0))
                if name not in self.p.inputs.seen:
                    self.p.inputs.seen.append(name)
        self.p.inputs.seen=self.p.inputs.seen[:8]
        remaining=[]
        self.p.builder.pending=[]
        for due,submitted,move,focus in self.pending:
            if due>=t:
                remaining.append((due,submitted,move,focus))
                self.p.builder.pending.append(NS(requested_tick=submitted))
            else:
                self.events['own',due,move.card]=dict(side='team',event_type='card_play',seconds=due/20,card=move.card,x=move.x,y=move.y)
                if focus is not None:
                    self.p.recent=(focus.id,t,focus.y)
                self.p.submitted_focus.pop(submitted,None)
        self.pending=remaining
        self.p.inputs.events=sorted(self.events.values(),key=lambda e:e['seconds'])[-128:]
        hand=[self.name(int(k)) for k in field('hand_card')]
        if any(c not in CARDS for c in hand):
            return 0
        features=np.zeros(58,dtype='float32')
        towers=[]
        for i,z in enumerate(TOWERS):
            prefix=('own_' if z.owner==0 else 'enemy_')+z.tower_kind.replace('princess_','')
            hp,maxhp=field(prefix+'_hp'),field(prefix+'_max_hp')
            towers.append(NS(owner=z.owner,tower_kind=z.tower_kind,active=hp>0,position=z.position,
                             hitpoints=hp,max_hitpoints=maxhp))
            features[i]=max(0,hp)/max(1,maxhp)
        balance=max(0,field('elixir')-field('elixir_reserved'))/10000
        features[6:10]=balance/10,self.p.inputs.enemy_elixir/10,min(1.,t/7200),min(1.,len(self.pending)/4)
        bins=features[10:].reshape(2,3,2,4)
        units=[]
        allies=[]
        for e in s.entities(v):
            val=lambda k:int(e[s.entity[k]])
            flags=val('flags')
            if val('kind')!=s.kinds['character'] or val('side') not in (0,1) or not flags&s.flags['alive']:
                continue
            if flags & (s.flags['underground']|s.flags['hidden']|s.flags['invisible']):
                continue
            structure=bool(flags&s.flags['building'])
            x,y=val('x'),val('y')
            if x<0 or y<0:
                continue
            raw_hp=val('hp')
            hp=raw_hp if raw_hp>=0 else None
            air=bool(flags&s.flags['air'])
            if not structure:
                bins[val('side'),min(2,y*3//32000),int(x>=9000)]+=(.2,(hp or 0)/6000,.2 if hp is None else 0.,.2 if air else 0)
            if val('side')==0 and (self.teacher or getattr(self.p,'match_strategy',None)) and (self.p.forecast or self.name(val('row')) in {'musketeer','cannon','ice-golem','hog-rider'}):
                allies.append(Unit(val('id'),None,self.name(val('row')),x,y,air,False,hp,structure=structure))
            if val('side')!=1 or (structure and not self.new):
                continue
            eid=val('id')
            name=self.name(val('row'))
            px,py,pt=self.seen.get(eid,(x,y,t-1))
            self.seen[eid]=(x,y,t)
            self.origins.setdefault(eid,y)
            units.append(Unit(eid,None,name,x,y,air,self.facts.get(name,{}).get('target')=='buildings',hp,
                              (x-px)/max(1,t-pt),(y-py)/max(1,t-pt),structure,None,self.origins[eid]))
        np.clip(bins,0,2,out=bins)
        self.p.allies=allies
        if t-self.last_decision<(5 if any(u.y<=21000 for u in units) else 20):
            return 0
        self.last_decision=t
        masks=self.arena.masks[self.i,self.seat,1:2305].reshape(4,32,18)
        obs=NS(tick=t,towers=towers,players=[NS(hand=hand,elixir_exact=balance)],
               action_mask=NS(reasons={'effective_elixir':balance},hand_slots=masks.any(axis=(1,2)),
                              placement_masks={str(i):{'row_major':m} for i,m in enumerate(masks)}))
        start=time.perf_counter()
        if self.teacher:
            move,focus=self.teacher.choose(self,obs,units,allies,hand,balance)
        else:
            move,focus=self.p.plan(obs,units,features)
        self.timings.append((time.perf_counter()-start)*1000)
        if self.trace is not None:
            self.trace.write(json.dumps(dict(tick=t,seat=self.seat,policy=self.i,elixir=balance,hand=hand,
                                             reason=move.reason,card=move.card,tile=[move.x//1000,move.y//1000],
                                             threats=[(u.name,u.x,u.y) for u in units],hp=[z.hitpoints for z in towers]))+'\n')
        if move.wait:
            return 0
        if not masks[move.slot,move.y//1000,move.x//1000]:
            raise RuntimeError('Illegal placement')
        self.pending.append((t+s.play_delay_ticks,t,move,focus))
        self.p.submitted_focus[t]=focus
        self.plays[move.card]+=1
        if focus:
            self.uses[focus.name]+=1
        return s.play(move.slot,move.x//1000,move.y//1000)


def main():
    import argparse
    args=argparse.ArgumentParser()
    args.add_argument('--out',type=Path,required=True)
    args.add_argument('--groups',type=int,default=4)
    args.add_argument('--start',type=int,default=827100)
    args.add_argument('--matchups',nargs='+',choices=opponents,default=list(opponents))
    args.add_argument('--trace',type=Path)
    args.add_argument('--match-strategy',type=Path)
    args.add_argument('--rollout-head',type=Path)
    args.add_argument('--public-classes',type=Path)
    args.add_argument('--fit-strategies',type=Path,help='Evaluate a JSON mapping of named parameter sets on common training games.')
    args.add_argument('--reference-bot',action='store_true',help='Include CRForge scripted control on the player deck for calibration.')
    args.add_argument('--teacher-data',type=Path)
    args.add_argument('--horizon',type=int,default=170)
    args.add_argument('--teacher-interval',type=int,default=30)
    args.add_argument('--teacher-objective',choices=('initial','balanced'),default='balanced')
    for name in ('reference','firstlight','simulator','tables','base','checkpoint','tile-prior','defense-responses','baseline-source'):
        args.add_argument('--'+name,type=Path,required=True)
    a=args.parse_args()
    limit_cpu()
    configure(a.firstlight,a.reference)
    sys.path.insert(0,str(a.simulator.resolve()/'python'))
    java=shutil.which('java')
    if java:
        os.environ.setdefault('JAVA_HOME',str(Path(java).resolve().parents[1]))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from crforge_env import VectorBattle,LadderConfig,ScriptedBot
    from memory_policy import MemoryPolicy
    catalog=json.loads((a.reference/'engine/data/cards.json').read_text(encoding='utf8'))
    facts={r['id'].replace('_','-'):r for r in catalog}
    native={k:r['sc_name'] for k,r in facts.items()}
    reverse={v:k for k,v in native.items()}
    trained=actor(MemoryPolicy,a)
    if a.fit_strategies:
        from match_strategy import MatchStrategy
        templates={}
        for name,params in json.loads(a.fit_strategies.read_text(encoding='utf8')).items():
            policy=copy.copy(trained)
            policy.match_strategy=MatchStrategy(**params)
            templates[name]=policy
    else:
        templates={'previous':actor(load_previous(a.baseline_source),a),'trained':trained}
    if a.teacher_data:
        templates={'teacher':trained}
    if a.reference_bot:
        templates['scripted']=trained
    names=list(templates)
    width=len(names)
    from contextlib import nullcontext
    with (a.teacher_data.open('xb') if a.teacher_data else nullcontext()) as data,(a.trace.open('x',encoding='utf8') if a.trace else nullcontext()) as trace, a.out.open('x',encoding='utf8') as out,VectorBattle(width,1,install=a.simulator/'env/build/install/env',tables=a.tables,
              config=LadderConfig(decision_ticks=10,level=11,tower_level=11,mask_version=2),
              jvm_args=('-Xmx256m','-XX:ActiveProcessorCount=1','-XX:+UseSerialGC')) as arena:
        bot=ScriptedBot()
        bot.bind(arena)
        for i in range(a.groups):
            for j,(matchup,deck) in enumerate(opponents.items()):
                if matchup not in a.matchups:
                    continue
                seed=a.start+i*100+j
                for seat in (0,1):
                    rng=random.Random(seed)
                    d=[native[n] for n in CARDS];e=[native[n] for n in deck]
                    rng.shuffle(d);rng.shuffle(e)
                    arena.reset([seed]*width,e if seat else d,d if seat else e)
                    players=[Player(arena,k,seat,p,hasattr(p,'responses'),facts,reverse) for k,p in enumerate(templates.values())]
                    for player in players:
                        player.trace=trace
                        if names[player.i]=='teacher':
                            from rollout_teacher import RolloutTeacher
                            player.teacher=RolloutTeacher(data,a.horizon,a.teacher_interval,a.teacher_objective)
                    randoms=[np.random.default_rng(seed) for _ in players]
                    script_rng=np.random.default_rng(seed)
                    done=set()
                    while len(done)<width:
                        arena.actions.fill(0)
                        for k,p in enumerate(players):
                            if k not in done:
                                arena.actions[k,seat]=(bot(arena.observations[k,seat],arena.masks[k,seat],script_rng)
                                                       if names[k]=='scripted' else p.action())
                                arena.actions[k,1-seat]=bot(arena.observations[k,1-seat],arena.masks[k,1-seat],randoms[k])
                        arena.step()
                        for k,p in enumerate(players):
                            if k in done:
                                continue
                            info=arena.info(k)
                            if info['fault'] or info['illegal_0'] or info['illegal_1'] or info['truncated']:
                                raise RuntimeError(str(info))
                            if not arena.dones[k]:
                                continue
                            done.add(k)
                            view=arena.observations[k,seat]
                            row=dict(seed=seed,seat=seat,matchup=matchup,policy=names[k],
                                     win=info['winner']==seat,draw=info['winner']==arena.schema.winner_draw,
                                     princess=sum(max(0,arena.schema.field(view,key)) for key in ('own_left_hp','own_right_hp')),
                                     enemy_princess=sum(max(0,arena.schema.field(view,key)) for key in ('enemy_left_hp','enemy_right_hp')),
                                     own_crowns=arena.schema.field(view,'crowns'),other_crowns=arena.schema.field(view,'crowns_enemy'),
                                     median_plan_ms=statistics.median(p.timings) if p.timings else None,
                                     p95_plan_ms=float(np.quantile(p.timings,.95)) if p.timings else None,
                                     forecast_error=p.p.forecast.error if getattr(p.p,'forecast',None) else None,
                                     actions=p.plays,defended=p.uses)
                            out.write(json.dumps(row)+'\n');out.flush()
            print('Finished seed group',i+1,flush=True)


if __name__=='__main__':
    main()
