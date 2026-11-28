import argparse
import json
from pathlib import Path
import statistics
import time

from runtime import ADB, REPO, account_id, bridge, configure


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
        log('ready',reader='NullsAI/native_runner',mode='observe' if args.observe_only else 'play',
            placement_input=None if args.observe_only else args.placement_input)
        if args.start_signal:
            while not args.start_signal.exists():
                if args.log.with_suffix('.STOP').exists():
                    return
                time.sleep(.2)
        attach_start=time.monotonic()
        if args.without_clock_cache:
            client.reader._clock_hint=None
        deadline=attach_start+args.wait
        arm_errors={}
        while True:
            if args.log.with_suffix('.STOP').exists():
                return
            probe_start=time.monotonic()
            quick=probe_start<deadline
            try:
                client.arm(account_id(args.player_tag),quick=quick)
                break
            except ExternalMemoryError as exc:
                reason=str(exc)
                arm_errors[reason]=arm_errors.get(reason,0)+1
                if arm_errors[reason]==1 or arm_errors[reason]%5==0:
                    log('waiting',reason=reason,attempt=arm_errors[reason])
                if not quick:
                    log('unavailable',reasons=arm_errors)
                    return
                time.sleep(min(.2,max(0.,deadline-time.monotonic())))
        armed_at = time.monotonic()
        log('armed', attach_ms=round((armed_at-probe_start)*1000,1),
            wait_before_lookup_ms=round((probe_start-attach_start)*1000,1), source=client.reader.discovery_source)
        cursor=0
        previous=-1
        generation=None
        count=0
        timings=[]
        last_advance=time.monotonic()
        deadline=time.monotonic()+(30 if args.observe_only else 370)
        while time.monotonic()<deadline and not args.log.with_suffix('.STOP').exists():
            start=time.monotonic()
            try:
                raw=client.observe(cursor)
            except ExternalMemoryError as exc:
                if time.monotonic()-last_advance>4:
                    log('unavailable',reason=str(exc)); return
                time.sleep(.2)
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
            dt=tick-previous if previous>=0 else 0
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
            last_advance=time.monotonic()
            cursor=raw['eventCursor']
            ms=(time.monotonic()-start)*1000
            timings.append(ms)
            count+=1
            if count==1:
                log('first_observation', from_lookup_ms=round((time.monotonic()-probe_start)*1000,1),
                    tick=tick)
            enemy=[e for e in obs.entities if e.owner==1 and e.entity_kind=='troop' and e.visible]
            own=raw['players'][0]
            log('snapshot',tick=tick,hand=own['hand'],elixir=own['elixir'],cycle=own['cycle'],next_card=own['nextCard'],
                enemies=[dict(id=e.entity_id,card=e.card_id,unit=e.native_data_global_id,xy=e.position,hp=e.hitpoints) for e in enemy],
                unknown_public_entities=[dict(id=e.entity_id,card=e.card_id,unit=e.native_data_global_id,xy=e.position)
                                         for e in obs.entities if e.visible and e.entity_kind=='unknown'],
                towers=[(t.owner,t.tower_kind,t.hitpoints) for t in obs.towers],snapshot_ms=round(ms,2))
            if obs.terminal.ended or (args.observe_only and count>=4 and enemy):
                if actor is not None:
                    actor.record_receipts(obs,log)
                log('complete',snapshots=count,median_snapshot_ms=round(statistics.median(timings),2),crowns=[p.crowns for p in obs.players]);return
            if actor is not None:
                if dt>0:
                    actor.step(raw,obs,cursor,log)
                else:
                    actor.inputs.update(raw,obs)
            interval=.32 if (builder.pending or any(e.position[1]<=21000 for e in enemy)) else .55
            time.sleep(max(0.,interval-(time.monotonic()-start)))
        log('ended',snapshots=count)


if __name__=='__main__':
    main()
