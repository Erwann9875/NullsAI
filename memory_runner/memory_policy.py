import hashlib
import math
import time

import numpy as np
import torch

from native_runner.contracts import ActionKind,ActionV1,TargetKind
from native_runner.live_probe.live_actions import AdbTouchController,LivePlayDispatcher,TouchLayout
from defensive_placement import aim_for, closest_defensive_tile
from bush_defense import BUSH_GOBLIN_ID, BushResponse
from defense_responses import (ResponseTable, Unit, building_name, can_hit, choose_counter,
                               fireball_target, threat_priority)
from live_decision import CARD_COST,DECK,GreedyPolicy,Move,tile_tables
from match_strategy import MatchStrategy, tile_move
from policy import SearchPolicy
from public_input import ReplayInput
from response_context import ContextHead
from runtime import REPO

BASE_SHA = 'e0b7a48db490dffe7a2b26423c40e332bae4822e00b3cfa32a3ebe15a7b2247b'
SELECTED_SHA = 'ef6614717c5337add5c16668dc6659cdfc6127fb6fdad4778e8cc559d9cd41ad'


def cannon_defense(obs, pressure, allies, hand, ready, elixir):
    """Reserve the building for a visible ground push and protect its cost from counterattacks."""
    if 'cannon' not in hand:
        return None, False
    exposed = []
    for unit in pressure:
        if unit.air is not False or unit.structure or unit.hp is None or unit.hp < 600:
            continue
        left = unit.x < 9000
        site_x = 7500 if left else 10500
        if any(a.name == 'cannon' and a.hp and a.hp > 0 and abs(a.x-site_x) <= 2500
               and 6000 <= a.y <= 14500 for a in allies):
            continue
        exposed.append(unit)
    protect_budget = bool(exposed) and elixir < CARD_COST['hog-rider'] + CARD_COST['cannon']
    interceptors = {'mega-knight','hog-rider','ram-rider','battle-ram','royal-hogs'}
    target = min((u for u in exposed if u.name in interceptors), key=lambda u:u.y, default=None)
    if target is None:
        return None, protect_budget
    if target.y > 17500 and elixir >= CARD_COST['cannon'] + max(CARD_COST.values()):
        return None, protect_budget
    if elixir < CARD_COST['cannon'] or target.y > 17500:
        return (Move(reason='reserve_cannon_for_push'), target), protect_budget
    if 'cannon' not in ready:
        return None, protect_budget
    if target.name == 'mega-knight':
        point = (7500 if target.x < 9000 else 10500, 8500)
    else:
        point = aim_for('cannon', target, [target], obs.towers)
    move = tile_move('cannon', ready, obs, point, 'intercept_ground_push')
    if move is None or math.dist((move.x,move.y),point) > 2500:
        return None, protect_budget
    return (move, target), protect_budget


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
        explicit=getattr(args,'defense_responses',None)
        path=explicit or args.tile_prior.with_name('defense-responses.npz')
        self.responses=ResponseTable(path) if explicit or path.exists() else None
        bush_path=args.tile_prior.with_name('bush-defense.npz')
        self.bush_response=BushResponse(bush_path) if bush_path.is_file() else None
        context_path=path.with_name('defense-context.npz')
        self.context=ContextHead(context_path) if self.responses and context_path.exists() else None
        self.first_y={}
        self.allies=[]
        explicit_strategy=getattr(args,'match_strategy',None)
        strategy_path=explicit_strategy or args.tile_prior.with_name('match-strategy.json')
        self.match_strategy=MatchStrategy.load(strategy_path) if explicit_strategy or strategy_path.exists() else None
        self.general_strategy=None
        if self.match_strategy and self.match_strategy.matchup_gate:
            import json
            general=json.loads(strategy_path.read_text(encoding='utf8')).get('general')
            if general:
                self.general_strategy=MatchStrategy(**general)
        from rollout_choices import RolloutHead
        head_path=getattr(args,'rollout_head',None) or args.tile_prior.with_name('rollout-head.npz')
        self.rollout_head=RolloutHead(head_path) if getattr(args,'rollout_head',None) or head_path.exists() else None
        if self.match_strategy and self.rollout_head is None and not getattr(args,'teacher_data',None):
            raise ValueError('The full-match policy requires rollout-head.npz next to the tile prior or --rollout-head.')
        self.next_rollout_tick=0
        self.forecast=None
        if getattr(args,'public_classes',None) or (self.match_strategy and self.match_strategy.search_width):
            args.public_classes=getattr(args,'public_classes',None) or args.tile_prior.parent/'crforge-public'
            args.simulator=getattr(args,'simulator',None) or args.firstlight/'.runtime/crforge'
            args.tables=getattr(args,'tables',None) or args.firstlight/'.runtime/crforge-cache/tables/16.635.8'
            if not (args.public_classes/'PublicBoard.class').is_file():
                raise ValueError('Compile PublicBoard.java and supply --public-classes or install it next to the tile prior in crforge-public.')
            from public_forecast import PublicForecast
            self.forecast=PublicForecast(args)
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
        allies=[]
        first_y={}
        for e in obs.entities:
            if not e.visible or e.owner not in (0,1) or e.entity_kind not in {'troop','building'} or e.hitpoints==0:
                continue
            key=(e.native_data_global_id,e.card_id)
            if key not in self.metadata:
                cat=self.builder.bundle.entity_archetype_catalog
                meta=cat.metadata_for_vocab_id(cat.runtime_global_vocab_id(e.native_data_global_id))
                spec=self.builder.bundle.card_specs.get(e.card_id)
                name = ('bush-goblin' if e.native_data_global_id==BUSH_GOBLIN_ID else
                        building_name(self.inputs.card(e.card_id) if spec else meta.archetype_key))
                self.metadata[key]=(name,
                                    meta.is_airborne if meta.is_airborne_known else None,
                                    bool(spec and spec.categorical_features.get('target_only_buildings') is True))
            name,air,buildings=self.metadata[key]
            x,y=map(int,e.position)
            structure=e.entity_kind=='building' or name in {'tesla','mortar','x-bow'}
            if not structure:
                bins[e.owner,min(2,y*3//32000),int(x>=9000)]+=(.2,(e.hitpoints or 0)/6000,.2 if e.hitpoints is None else 0.,.2 if air else 0.)
            if e.owner==1:
                vx, vy = e.velocity or (0., 0.)
                first_y[e.entity_id]=self.first_y.get(e.entity_id,y)
                threats.append(Unit(e.entity_id,e.card_id,name,x,y,air,buildings,e.hitpoints,vx,vy,
                                    structure,getattr(e,'targetable',None),first_y[e.entity_id]))
            elif name=='cannon' or (self.match_strategy and (self.forecast or name in {'musketeer','ice-golem','hog-rider'})):
                allies.append(Unit(e.entity_id,e.card_id,name,x,y,air,buildings,e.hitpoints,structure=structure))
        self.first_y=first_y
        self.allies=allies
        np.clip(bins,0,2,out=bins)
        return threats,features

    def plan(self,obs,units,features):
        self.last_planner='base'
        own=obs.players[0]
        hand=[self.inputs.card(i) for i in own.hand]
        elixir=min(own.elixir_exact, float(obs.action_mask.reasons['effective_elixir']))
        if self.elixir_cap is not None:
            elixir=min(elixir,self.elixir_cap)
        ready={c:i for i,c in enumerate(hand) if CARD_COST[c]<=elixir and obs.action_mask.hand_slots[i]}
        def legal(card):
            area=obs.action_mask.placement_masks[str(ready[card])]['row_major']
            return np.asarray(area,dtype=bool)[self.rows,self.cols]&self.tile_masks[card in ('fireball','the-log')].numpy()
        priority={u.id:threat_priority(u,obs.towers) for u in units}
        pressure=[u for u in units if math.isfinite(priority[u.id])]
        if self.builder.pending:
            if len(self.builder.pending) >= 2:
                return Move(reason='awaiting_deployment'),None
            for pending in self.builder.pending:
                target = self.submitted_focus.get(pending.requested_tick)
                if target:
                    pressure = [u for u in pressure if (target.x < 9000) != (u.x < 9000)]
            if not pressure:
                return Move(reason='awaiting_deployment'),None
        lead=min(pressure,key=lambda u:priority[u.id],default=None)
        if lead and lead.name=='bush-goblin' and self.bush_response is not None:
            group=[u for u in pressure if u.name==lead.name and abs(u.x-lead.x)<=3500 and abs(u.y-lead.y)<=3000]
            if self.recent and any(u.id==self.recent[0] for u in group) and obs.tick-self.recent[1]<28:
                return Move(reason='defender_recently_deployed'),lead
            move=self.bush_response.choose(obs,group,ready)
            if move is not None:
                self.last_planner='bush_response'
                return move,lead
        intercept, protect_budget = cannon_defense(obs, pressure, self.allies, hand, ready, elixir)
        if intercept is not None:
            self.last_planner='cannon_defense'
            return intercept
        strategy=self.match_strategy
        if strategy and strategy.matchup_gate:
            seen=set(self.inputs.seen).union(u.name for u in units)
            if not seen.intersection(strategy.specialist_triggers):
                strategy=(self.general_strategy if not strategy.general_triggers or seen.intersection(strategy.general_triggers) else None)
        if strategy and self.rollout_head and strategy.rollout_weight>=1 and (
                self.builder.pending or obs.tick<self.next_rollout_tick):
            return Move(reason='rollout_wait'),None
        if strategy and not self.builder.pending:
            if self.rollout_head and strategy.rollout_weight and ready and obs.tick>=self.next_rollout_tick:
                from rollout_choices import choices
                anchors=[a for a in self.allies if a.name in {'cannon','musketeer','hog-rider','ice-golem'}]
                rollout_ready = {c:i for c,i in ready.items() if c!='hog-rider' or not protect_budget}
                moves,rows=choices(obs,units,anchors,rollout_ready,elixir)
                scores=self.rollout_head.scores(rows)
                self.last_planner='distilled'
                if self.forecast and strategy.search_width:
                    scores=self.forecast.score(obs,units,self.allies,moves,scores,strategy,elixir,self.inputs.events)
                    if not self.forecast.error:
                        self.last_planner='forecast'
                scores[0]=0.
                scores[1:]-=strategy.rollout_threshold
                for i,m in enumerate(moves[1:],1):
                    scores[i]-=strategy.attack_margin if m.card=='hog-rider' else strategy.defensive_margin
                    if pressure and m.y>=14000 and m.card not in {'fireball','the-log'}:
                        scores[i]-=strategy.near_tower_margin
                choice=int(scores.argmax())
                if choice or strategy.rollout_weight>=1:
                    self.next_rollout_tick=obs.tick+strategy.rollout_interval
                    return moves[choice],None
                self.last_planner='base'
        if pressure:
            u=min(pressure,key=lambda t:priority[t.id])
            lane=int(u.x>=9000)
            if self.recent and self.recent[0]==u.id and obs.tick-self.recent[1]<28 and u.y>=self.recent[2]-1500:
                return Move(reason='defender_recently_deployed'),u
            cluster=[t for t in units if can_hit(t) and (u.x-t.x)**2+(u.y-t.y)**2<=2700**2]
            card_prior,tiles,hidden=(self.responses.contextual(u,self.context,self.inputs.events[-32:],hand,obs.tick)
                                     if self.responses else (None,None,None))
            spell_point,spell_value=None,0.
            if 'fireball' in ready:
                spell_point,spell_value=fireball_target(obs.action_mask.placement_masks[str(ready['fireball'])]['row_major'],
                                                       u,units,obs.towers)
            card=choose_counter(ready,u,cluster,card_prior,spell_value)
            if card is None:
                return Move(reason='save_elixir_for_defense'),u
            if card=='fireball':
                point=spell_point
            else:
                aim=aim_for(card,u,cluster,obs.towers)
                if self.responses:
                    aim=self.responses.adjust_aim(aim,u,card)
                rows=obs.action_mask.placement_masks[str(ready[card])]['row_major']
                index=DECK.index(card)
                prior=self.prior[index,1+lane if u.buildings and u.air is False else 0]
                learned=tiles[index] if tiles is not None else None
                if hidden is not None:
                    surface=self.context.tiles(hidden,card,u.x>=9000)
                    learned=surface if learned is None else .5*(learned+surface)
                point=closest_defensive_tile(card,rows,aim,prior,obs.towers,
                                             learned if card!='the-log' else None)
            if point is None:
                return Move(reason='no_legal_tile'),u
            x,y=point
            return Move(False,ready[card],card,x,y,'defend_siege' if u.structure else 'defend_memory_troop',(lane,u.y)),u
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
        defending=any(e.owner==1 and e.visible and e.entity_kind in {'troop','building'} and e.position[1]<=21000 for e in obs.entities)
        if obs.tick-self.last_decision<(5 if defending else 20):
            return
        self.last_decision=obs.tick
        begin=time.perf_counter()
        units,features=self.read_units(obs)
        move,focus=self.plan(obs,units,features)
        ms=(time.perf_counter()-begin)*1000
        log('decision',tick=obs.tick,hand=[self.inputs.card(c) for c in obs.players[0].hand],elixir=obs.players[0].elixir_exact,
            spendable_elixir=round(self.elixir_cap,3),
            planner=self.last_planner,forecast_error=self.forecast.error if self.forecast else None,
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
