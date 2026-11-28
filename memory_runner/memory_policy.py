from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch

from native_runner.contracts import ActionKind,ActionV1,TargetKind
from native_runner.live_probe.live_actions import AdbTouchController,LivePlayDispatcher,TouchLayout
from defensive_placement import aim_for, closest_defensive_tile, projected
from live_decision import CARD_COST,DECK,GreedyPolicy,Move,tile_tables
from policy import SearchPolicy
from public_input import ReplayInput
from runtime import REPO

BASE_SHA = 'e0b7a48db490dffe7a2b26423c40e332bae4822e00b3cfa32a3ebe15a7b2247b'
SELECTED_SHA = 'ef6614717c5337add5c16668dc6659cdfc6127fb6fdad4778e8cc559d9cd41ad'


@dataclass(frozen=True)
class Unit:
    id:int
    card_id:int|None
    name:str
    x:int
    y:int
    air:bool|None
    buildings:bool
    hp:float|None
    vx:float = 0.
    vy:float = 0.


class MemoryPolicy:
    def __init__(self,builder,args,controller=None):
        base=args.base
        selected=args.checkpoint
        if base.name != 'best_model.pt':
            raise ValueError('--base must point to best_model.pt')
        for path, expected in ((base, BASE_SHA), (selected, SELECTED_SHA)):
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Unexpected checkpoint at {path}')
        model=SearchPolicy(model_dir=base.parent,device='cpu',tower_hp=False)
        model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=True)['policy'])
        self.fast=GreedyPolicy(model)
        self.inputs=ReplayInput(model,builder.bundle.card_specs)
        self.builder=builder
        if controller is None:
            layout=TouchLayout.load(REPO/'native_runner/live_probe/mumu_900x1600.json')
            controller=AdbTouchController(args.adb,args.serial,layout,placement_input=args.placement_input)
            controller.verify()
        self.dispatch=LivePlayDispatcher(builder,controller)
        self.points,self.tile_masks=tile_tables()
        self.cols,self.rows=(self.points//1000).T
        with np.load(args.tile_prior, allow_pickle=False) as data:
            if tuple(data['cards'])!=DECK or data['log_probs'].shape!=(8,3,18,32):
                raise ValueError('Placement prior card order differs')
            self.prior=data['log_probs'].reshape(8,3,576)
        self.last_decision=-20
        self.submitted_focus={}
        self.recent=None
        self.metadata={}
        self.deployments=[]
        self.elixir_cap=None
        self.elixir_tick=0
        self.balance_settles_at=0

    def update_elixir(self, obs):
        raw = obs.players[0].elixir_exact
        if self.elixir_cap is None or (not self.builder.pending and obs.tick >= self.balance_settles_at):
            self.elixir_cap = raw
        else:
            elapsed = max(0, obs.tick - self.elixir_tick)
            regenerated = elapsed * .05 * obs.time.elixir_multiplier / 2.8
            self.elixir_cap = min(raw, self.elixir_cap + regenerated)
        self.elixir_tick = obs.tick

    def observe_deployments(self, obs, log):
        remaining = []
        for tick, card, grid, previous_ids in self.deployments:
            spawned = [e for e in obs.entities if e.owner == 0 and e.card_id == card and
                       e.entity_id not in previous_ids and e.entity_kind in {'building', 'troop'}
                       and e.hitpoints is not None and e.hitpoints > 0]
            if spawned:
                first = min(spawned, key=lambda e: e.entity_id)
                log('deployment_observed', tick=obs.tick, requested_tick=tick, card=self.inputs.card(card),
                    target_grid=grid, position=first.position, kind=first.entity_kind)
            elif obs.tick - tick <= 60:
                remaining.append((tick, card, grid, previous_ids))
        self.deployments = remaining

    def read_units(self,obs):
        features=np.zeros(58,dtype=np.float32)
        for t in obs.towers:
            features[t.owner*3+('king','princess_left','princess_right').index(t.tower_kind)]=t.hitpoints/max(1.,t.max_hitpoints)
        features[6:10]=(obs.players[0].elixir_exact/10,self.inputs.enemy_elixir/10,min(1.,obs.tick/7200),min(1.,len(self.builder.pending)/4))
        bins=features[10:].reshape(2,3,2,4)
        threats=[]
        for e in obs.entities:
            if not e.visible or e.owner not in (0,1) or e.entity_kind!='troop' or e.hitpoints==0:
                continue
            key=(e.native_data_global_id,e.card_id)
            if key not in self.metadata:
                cat=self.builder.bundle.entity_archetype_catalog
                meta=cat.metadata_for_vocab_id(cat.runtime_global_vocab_id(e.native_data_global_id))
                spec=self.builder.bundle.card_specs.get(e.card_id)
                self.metadata[key]=(self.inputs.card(e.card_id) if spec else meta.archetype_key,
                                    meta.is_airborne if meta.is_airborne_known else None,
                                    bool(spec and spec.categorical_features.get('target_only_buildings') is True))
            name,air,buildings=self.metadata[key]
            x,y=map(int,e.position)
            bins[e.owner,min(2,y*3//32000),int(x>=9000)]+=(.2,(e.hitpoints or 0)/6000,.2 if e.hitpoints is None else 0.,.2 if air else 0.)
            if e.owner==1:
                vx, vy = e.velocity or (0., 0.)
                threats.append(Unit(e.entity_id,e.card_id,name,x,y,air,buildings,e.hitpoints,vx,vy))
        np.clip(bins,0,2,out=bins)
        return threats,features

    def plan(self,obs,units,features):
        own=obs.players[0]
        hand=[self.inputs.card(i) for i in own.hand]
        elixir=min(own.elixir_exact, float(obs.action_mask.reasons['effective_elixir']))
        if self.elixir_cap is not None:
            elixir=min(elixir,self.elixir_cap)
        ready={c:i for i,c in enumerate(hand) if CARD_COST[c]<=elixir and obs.action_mask.hand_slots[i]}
        def legal(card):
            area=obs.action_mask.placement_masks[str(ready[card])]['row_major']
            return np.asarray(area,dtype=bool)[self.rows,self.cols]&self.tile_masks[card in ('fireball','the-log')].numpy()
        pressure=[u for u in units if u.y<=(21000 if u.buildings else 17800)]
        if self.builder.pending:
            if len(self.builder.pending) >= 2:
                return Move(reason='awaiting_deployment'),None
            for pending in self.builder.pending:
                target = self.submitted_focus.get(pending.requested_tick)
                if target:
                    pressure = [u for u in pressure if (target.x < 9000) != (u.x < 9000)]
            if not pressure:
                return Move(reason='awaiting_deployment'),None
        if pressure:
            u=min(pressure,key=lambda t:t.y-(1800 if t.buildings else 0))
            lane=int(u.x>=9000)
            if self.recent and self.recent[0]==u.id and obs.tick-self.recent[1]<28 and u.y>=self.recent[2]-1500:
                return Move(reason='defender_recently_deployed'),u
            cluster=[t for t in pressure if (u.x-t.x)**2+(u.y-t.y)**2<=2700**2]
            if len(cluster)>=3 and 'the-log' in ready and all(t.air is False and t.hp is not None and t.hp<=420 for t in cluster):
                card='the-log'
            elif len(cluster)>=2 and 'fireball' in ready and (len(cluster)>=3 or sum(t.hp or 0 for t in cluster)>900):
                card='fireball'
            elif u.buildings and u.air is False and u.y>=7500 and 'cannon' in ready:
                card='cannon'
            elif 'musketeer' in ready:
                card='musketeer'
            elif u.air is False and 'cannon' in ready and u.y>=8500:
                card='cannon'
            elif 'ice-spirit' in ready and projected(u,18)[1]<12500 and u.name!='electro-giant':
                card='ice-spirit'
            elif 'skeletons' in ready and u.air is False and not u.buildings and projected(u,18)[1]<11000:
                card='skeletons'
            else:
                return Move(reason='save_elixir_for_defense'),u
            aim=aim_for(card,u,cluster,obs.towers)
            rows=obs.action_mask.placement_masks[str(ready[card])]['row_major']
            prior=self.prior[DECK.index(card),1+lane if u.buildings and u.air is False else 0]
            point=closest_defensive_tile(card,rows,aim,prior,obs.towers)
            if point is None:
                return Move(reason='no_legal_tile'),u
            x,y=point
            return Move(False,ready[card],card,x,y,'defend_memory_troop',(lane,u.y)),u
        allowed={c for c in ready if c not in {'cannon','the-log','fireball'} and elixir-CARD_COST[c]>=4 and elixir>=(8 if c=='hog-rider' else 9)}
        px,py=self.points.T
        masks={c:legal(c) for c in allowed}
        if 'hog-rider' in masks:
            masks['hog-rider']&=(py>=12500)&(py<15000)&((abs(px-3700)<2100)|(abs(px-14300)<2100))
        allowed={c for c in allowed if masks[c].any()}
        if not allowed:
            return Move(reason='hold_defensive_reserve'),None
        def scores(card):
            return torch.from_numpy(np.where(masks[card],.65*self.prior[DECK.index(card),0],-np.inf))
        return self.fast.choose(hand,elixir,obs.tick/20,self.inputs.events[-128:],features,allowed_cards=allowed,
                                tile_scores=scores,seen=self.inputs.seen[:8],enemy_elixir=self.inputs.enemy_elixir),None

    def record_receipts(self,obs,log):
        for item,status in self.builder.receipts:
            focus=self.submitted_focus.pop(item.requested_tick,None)
            if status == 'confirmed':
                self.balance_settles_at = obs.tick + 8
            log('receipt',tick=obs.tick,requested_tick=item.requested_tick,status=status,card=self.inputs.card(item.action.card_id),threat_id=focus.id if focus else None)
            if focus and status=='confirmed':
                self.recent=(focus.id,obs.tick,focus.y)

    def step(self,raw,obs,cursor,log):
        self.inputs.update(raw,obs)
        self.observe_deployments(obs, log)
        self.record_receipts(obs,log)
        self.update_elixir(obs)
        defending=any(e.owner==1 and e.visible and e.entity_kind=='troop' and e.position[1]<=21000 for e in obs.entities)
        if obs.tick-self.last_decision<(5 if defending else 20):
            return
        self.last_decision=obs.tick
        begin=time.perf_counter()
        units,features=self.read_units(obs)
        move,focus=self.plan(obs,units,features)
        ms=(time.perf_counter()-begin)*1000
        log('decision',tick=obs.tick,hand=[self.inputs.card(c) for c in obs.players[0].hand],elixir=obs.players[0].elixir_exact,
            spendable_elixir=round(self.elixir_cap,3),
            reason=move.reason,card=move.card,target=None if move.wait else [move.x,move.y],threat=vars(focus) if focus else None,inference_ms=round(ms,3))
        if move.wait:
            return
        own=obs.players[0]
        action=ActionV1(owner=0,kind=ActionKind.PLAY_CARD,hand_slot=move.slot,card_id=own.hand[move.slot],
                        target_kind=TargetKind.GRID,target_grid=(move.x//1000,move.y//1000))
        sending = time.perf_counter()
        self.dispatch.submit(obs,cursor,(action,),None)
        self.elixir_cap -= CARD_COST[move.card]
        log('touch_sent', tick=obs.tick, card=move.card, target_grid=action.target_grid,
            dispatch_ms=round((time.perf_counter()-sending)*1000,1))
        if move.card not in {'the-log', 'fireball'}:
            ids = {e.entity_id for e in obs.entities if e.owner == 0 and e.card_id == action.card_id}
            self.deployments.append((obs.tick, action.card_id, action.target_grid, ids))
        self.submitted_focus[obs.tick]=focus
        self.inputs.targets[own.hand[move.slot]]=(move.x,move.y)
