import argparse
import json
from pathlib import Path
import statistics
import time

from runtime import ADB, REPO, account_id, bridge, configure


def quick_lookup_seconds(wait, kickoff_signaled, indexed=False):
    return min(20.,wait) if kickoff_signaled and not indexed else wait


def observe_stable_frame(client, cursor, deadline, stop_file, log):
    from native_runner.live_probe.memory_types import InconsistentBattleFrame
    failures = 0
    started = time.monotonic()
    while True:
        try:
            frame = client.observe(cursor)
        except InconsistentBattleFrame as exc:
            failures += 1
            if failures == 1 or failures % 10 == 0:
                log('waiting_for_coherent_frame', attempts=failures, reason=str(exc))
            remaining = deadline - time.monotonic()
            if remaining <= 0 or stop_file.exists():
                return None
            time.sleep(min(.1 * failures, .5, remaining))
            if time.monotonic() >= deadline or stop_file.exists():
                return None
            continue
        if failures:
            log('coherent_frame_restored', tick=frame['tick'], attempts=failures,
                elapsed_ms=round((time.monotonic()-started)*1000,1))
        return frame


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--player-tag',required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--wait',type=float,default=65.)
    p.add_argument('--observe-only',action='store_true')
    p.add_argument('--placement-input', choices=('drag','tap'), default='drag',
                   help='Drag to the target tile (default) or tap the card and tile separately.')
    p.add_argument('--without-clock-cache',action='store_true')
    p.add_argument('--start-signal',type=Path)
    p.add_argument('--serial',default='127.0.0.1:16416')
    p.add_argument('--adb',type=Path,default=ADB)
    p.add_argument('--port',type=int,default=26842)
    p.add_argument('--firstlight',type=Path,default=REPO/'.runtime/firstlight')
    p.add_argument('--reference',type=Path,default=REPO/'.runtime/reference')
    p.add_argument('--base',type=Path,default=REPO/'.runtime/models/best_model.pt')
    p.add_argument('--checkpoint',type=Path,default=REPO/'.runtime/models/selected.pt')
    p.add_argument('--tile-prior',type=Path,default=REPO/'.runtime/models/tile-prior.npz')
    p.add_argument('--defense-responses',type=Path,help='Opponent-specific replay response table (default: next to --tile-prior).')
    p.add_argument('--match-strategy',type=Path,help='Full-match parameters (default: next to --tile-prior).')
    p.add_argument('--rollout-head',type=Path,help='Distilled CPU action values learned from simulator counterfactuals.')
    p.add_argument('--public-classes',type=Path)
    p.add_argument('--simulator',type=Path)
    p.add_argument('--tables',type=Path)
    args=p.parse_args()
    configure(args.firstlight, None if args.observe_only else args.reference)
    from support import limit_cpu
    limit_cpu()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from native_runner.live_probe.live_observation import LiveObservationBuilder
    from native_runner.live_probe.external_memory import ExternalMemoryError
    from native_runner.live_probe.live_contracts import LiveLadderError
    builder=LiveObservationBuilder(allow_unknown_entities=True)
    costs={i:int(spec.elixir_cost) for i,spec in builder.bundle.card_specs.items() if spec.elixir_cost is not None}
    actor=None
    if not args.observe_only:
        from memory_policy import MemoryPolicy
        actor=MemoryPolicy(builder,args)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('x',encoding='utf-8') as stream, bridge(args,costs) as client:
        def log(event,**fields):
            row = json.dumps(dict(event=event, **fields))
            print(row, file=stream, flush=True)
            if event in {'ready', 'armed', 'first_observation', 'complete', 'unavailable', 'ended'}:
                print(row, flush=True)
        if args.without_clock_cache:
            client.reader._clock_hint=None
        preparation = time.monotonic()
        prepare = args.start_signal or not getattr(client.reader, '_clock_hint', None)
        candidates = client.reader.prepare_kickoff() if prepare else 0
        log('ready',reader='NullsAI/native_runner',mode='observe' if args.observe_only else 'play',
            lobby_candidates=candidates,preparation_ms=round((time.monotonic()-preparation)*1000,1),
            placement_input=None if args.observe_only else args.placement_input,
            replay_defenses=actor is not None and actor.responses is not None,
            bush_defense=actor is not None and actor.bush_response is not None,
            context_defenses=actor is not None and actor.context is not None,
            match_strategy=actor is not None and actor.match_strategy is not None,
            rollout_head=actor is not None and actor.rollout_head is not None,
            public_forecast=actor is not None and actor.forecast is not None)
        if args.start_signal:
            while not args.start_signal.exists():
                if args.log.with_suffix('.STOP').exists():
                    return
                time.sleep(.05)
        attach_start=time.monotonic()
        deadline=attach_start+args.wait
        quick_until=attach_start+quick_lookup_seconds(args.wait,args.start_signal is not None,indexed=bool(candidates))
        full_attempts=0
        arm_errors={}
        while True:
            if args.log.with_suffix('.STOP').exists():
                return
            probe_start=time.monotonic()
            quick=probe_start<quick_until
            if not quick:
                full_attempts+=1
                log('full_lookup',attempt=full_attempts,elapsed_ms=round((probe_start-attach_start)*1000,1))
            try:
                layout=client.arm(account_id(args.player_tag),quick=quick)
                break
            except ExternalMemoryError as exc:
                reason=str(exc)
                arm_errors[reason]=arm_errors.get(reason,0)+1
                if arm_errors[reason]==1 or arm_errors[reason]%5==0:
                    log('waiting',reason=reason,attempt=arm_errors[reason])
                if not quick:
                    if not args.start_signal or full_attempts>=2 or time.monotonic()>=deadline:
                        log('unavailable',reasons=arm_errors)
                        return
                    quick_until=min(deadline,time.monotonic()+1.)
                time.sleep(min(.06 if args.start_signal else .2,max(0.,deadline-time.monotonic())))
        armed_at = time.monotonic()
        log('armed', attach_ms=round((armed_at-probe_start)*1000,1),
            from_start_ms=round((armed_at-attach_start)*1000,1),
            clock_hint_disabled=args.without_clock_cache,
            wait_before_lookup_ms=round((probe_start-attach_start)*1000,1), source=client.reader.discovery_source,
            native_local_owner=layout.native_local_owner)
        cursor=0
        previous=-1
        generation=None
        count=0
        timings=[]
        last_advance=time.monotonic()
        waiting_for_hand=False
        deadline=time.monotonic()+(30 if args.observe_only else 370)
        stop_file=args.log.with_suffix('.STOP')
        while time.monotonic()<deadline and not stop_file.exists():
            start=time.monotonic()
            try:
                raw=observe_stable_frame(client,cursor,deadline,stop_file,log)
                if raw is None:
                    break
            except ExternalMemoryError as exc:
                hand_error=str(exc) in {'local hand vector is not readable','local cycle vector is not readable'}
                if hand_error and not waiting_for_hand:
                    log('waiting_for_hand',reason=str(exc))
                    waiting_for_hand=True
                if time.monotonic()-last_advance>(20 if hand_error else 4):
                    log('unavailable',reason=str(exc)); return
                if count==0:
                    log('opening_retry',reason=str(exc))
                time.sleep(.35 if hand_error else (.04 if count==0 else .2))
                continue
            tick=raw['tick']
            if raw['players'][0]['accountId']!=account_id(args.player_tag) or (generation is not None and raw['generation']!=generation) or tick<previous:
                raise ValueError('Battle identity or clock changed')
            generation=raw['generation']
            if tick==previous:
                if time.monotonic()-last_advance>4:
                    log('unavailable',reason='frozen battle clock'); return
                time.sleep(.2)
                continue
            read_ms=(time.monotonic()-start)*1000
            try:
                obs=builder.build(raw)
            except LiveLadderError as exc:
                if str(exc) not in {'a public tower has no exact HP','the first live frame does not contain all six towers'}:
                    log('unavailable',reason=str(exc),tick=tick)
                    raise
                if time.monotonic()-last_advance>4:
                    log('unavailable',reason=str(exc),tick=tick)
                    return
                time.sleep(.12)
                continue
            previous=tick
            if waiting_for_hand:
                log('hand_restored',tick=tick)
                waiting_for_hand=False
            last_advance=time.monotonic()
            cursor=raw['eventCursor']
            ms=(time.monotonic()-start)*1000
            timings.append(ms)
            count+=1
            if count==1:
                log('first_observation', from_lookup_ms=round((time.monotonic()-probe_start)*1000,1),
                    read_ms=round(read_ms,1),build_ms=round(ms-read_ms,1),
                    from_start_ms=round((time.monotonic()-attach_start)*1000,1),tick=tick)
            enemy=[e for e in obs.entities if e.owner==1 and e.entity_kind in {'troop','building'} and e.visible]
            own=raw['players'][0]
            log('snapshot',tick=tick,hand=own['hand'],elixir=own['elixir'],cycle=own['cycle'],next_card=own['nextCard'],
                enemies=[dict(id=e.entity_id,card=e.card_id,unit=e.native_data_global_id,kind=e.entity_kind,xy=e.position,hp=e.hitpoints) for e in enemy],
                unknown_public_entities=[dict(id=e.entity_id,card=e.card_id,unit=e.native_data_global_id,xy=e.position)
                                         for e in obs.entities if e.visible and e.entity_kind=='unknown'],
                towers=[(t.owner,t.tower_kind,t.hitpoints) for t in obs.towers],snapshot_ms=round(ms,2))
            if obs.terminal.ended or (args.observe_only and count>=4 and enemy):
                if actor is not None:
                    actor.record_receipts(obs,log)
                log('complete',snapshots=count,median_snapshot_ms=round(statistics.median(timings),2),
                    crowns=[p.crowns for p in obs.players],opponent_cards=list(actor.inputs.seen) if actor else None);return
            if actor is not None:
                actor.step(raw,obs,cursor,log)
            interval=.32 if (builder.pending or any(e.position[1]<=21000 for e in enemy)) else .55
            time.sleep(max(0.,interval-(time.monotonic()-start)))
        log('ended',snapshots=count)


if __name__=='__main__':
    main()
