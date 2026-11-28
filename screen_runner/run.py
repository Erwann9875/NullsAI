"""One-thread, screen-only runner for Null's Royale."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

BASE_SHA = 'e0b7a48db490dffe7a2b26423c40e332bae4822e00b3cfa32a3ebe15a7b2247b'
SELECTED_SHA = 'ef6614717c5337add5c16668dc6659cdfc6127fb6fdad4778e8cc559d9cd41ad'


def require_checkpoint(path, expected):
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected checkpoint at {path}')

HERE = Path(__file__).resolve().parent
HAND_X = (282,451,619,787)
HAND_Y = 1425
CARD_COST = {'hog-rider':4,'musketeer':4,'ice-golem':2,'skeletons':1,
             'ice-spirit':1,'cannon':3,'fireball':4,'the-log':2}
SPELLS = {'fireball','the-log'}


def frame(adb):
    from PIL import Image
    payload = subprocess.check_output(adb+['exec-out','screencap'],timeout=5)
    width,height,pixel_format = struct.unpack_from('<3I',payload)
    if (width,height,pixel_format)!=(900,1600,1) or len(payload)!=width*height*4+16:
        raise RuntimeError('Expected the calibrated 900 by 1600 MuMu display')
    return Image.frombytes('RGBA',(width,height),payload[16:]).convert('RGB')


def patch(image, slot):
    from PIL import Image
    if image.size != (900,1600):
        image = image.resize((900,1600),Image.Resampling.BILINEAR)
    center = HAND_X[slot]
    crop = image.crop((center-40,1375,center+40,1470)).resize((20,24),Image.Resampling.BILINEAR)
    return signature(crop)


def signature(crop):
    import numpy as np
    rgb = np.asarray(crop,dtype=np.float32)
    flat = (rgb-rgb.mean(axis=(0,1))).ravel()
    return flat/max(1.,float(np.linalg.norm(flat)))


def recognizer():
    import numpy as np
    from PIL import Image
    entries = []
    for path in sorted((HERE/'templates').glob('*.png')):
        with Image.open(path) as sample:
            entries.append((path.stem.rsplit('-',1)[0], signature(sample.convert('RGB'))))
    if {name for name, _ in entries} != set(CARD_COST):
        raise ValueError('Missing card calibration templates')
    bank = np.stack([sample for _,sample in entries])

    def recognize(image):
        scores = bank @ np.stack([patch(image,i) for i in range(4)]).T
        hand = []
        quality = []
        for column in scores.T:
            by_card = {card:max(float(column[i]) for i,(name,_) in enumerate(entries) if name==card) for card in CARD_COST}
            top, second = sorted(by_card,key=by_card.get,reverse=True)[:2]
            score = by_card[top]
            hand.append(top if score>=.62 and score-by_card[second]>=.12 else None)
            quality.append(round(score,2))
        return hand,quality
    return recognize


def elixir_value(image):
    import numpy as np
    strip = np.asarray(image.crop((257,1550,867,1564)),dtype=np.int16)
    r,g,b = strip.transpose(2,0,1)
    filled = ((r>115) & (b>100) & (r>g+50) & (b>g+45)).mean(axis=0)>.5
    if not filled.any():
        return 0.
    last = int(np.flatnonzero(filled)[-1])
    if last>100 and filled[65:last+1].mean()<.72:
        return None
    visible = min(10.,(last+1)/61.)
    next_mark = math.ceil(visible)
    # The black divider at a whole-elixir mark obscures a few filled pixels.
    return float(next_mark) if next_mark-visible <= .13 else visible


def battlefield(image):
    # The timer is the stable black HUD box in the top-right corner. The hand
    # detector supplies an independent check before any drag is permitted.
    pixel = image.getpixel((747,45))
    return max(pixel)<36


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('original','selected','verify'),required=True)
    parser.add_argument('--match-kind',choices=('practice','normal'),default='practice')
    parser.add_argument('--legacy-placement',action='store_true',help='Use the old policy placement for comparison')
    parser.add_argument('--serial',default='127.0.0.1:16416')
    parser.add_argument('--log',type=Path)
    parser.add_argument('--record-board',action='store_true',help='Save a small board image every four seconds and before defensive plays')
    parser.add_argument('--seconds',type=float,default=355.)
    parser.add_argument('--reference',type=Path,required=True,help='Local pinned clash-royale-ai source checkout')
    parser.add_argument('--base',type=Path,required=True,help='Path to best_model.pt')
    parser.add_argument('--checkpoint',type=Path,required=True,help='Path to selected.pt')
    parser.add_argument('--adb',default='adb',help='ADB executable, used for live matches only')
    parser.add_argument('--frame',type=Path,help='Saved 900x1600 battle screenshot for verify mode')
    args=parser.parse_args()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    if args.mode!='verify' and args.log is None:
        parser.error('Live matches require --log')
    reference=args.reference.resolve()
    sys.path[:0]=[str(x) for x in (reference/'src',reference/'engine')]
    from support import limit_cpu, memory
    limits=limit_cpu()
    import torch
    from live_decision import GreedyPolicy, defense_move, features_from_threats, visible_threats
    from policy import SearchPolicy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, checkpoint = args.base, args.checkpoint
    if base.name != 'best_model.pt':
        parser.error('The upstream loader requires the base filename best_model.pt')
    require_checkpoint(base,BASE_SHA)
    require_checkpoint(checkpoint,SELECTED_SHA)
    policy=SearchPolicy(model_dir=base.parent,device='cpu',tower_hp=False)
    policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['policy'],strict=True)
    if args.mode=='original':
        policy.set_vector(torch.zeros_like(policy.card_adjustments))
    fast_policy=GreedyPolicy(policy)
    recognize=recognizer()
    adb=[args.adb,'-s',args.serial]
    if args.mode=='verify':
        from PIL import Image
        if args.frame:
            with Image.open(args.frame) as frame_image:
                image=frame_image.convert('RGB')
            if image.size!=(900,1600) or not battlefield(image):
                parser.error('Expected a 900x1600 active battle frame')
            hand,confidence=recognize(image)
            elixir=elixir_value(image)
            if None in hand or len(set(hand))!=4 or elixir is None:
                parser.error('Could not read the Hog cycle hand and elixir in this frame')
            threats=visible_threats(image)
        else:
            hand=['musketeer','fireball','hog-rider','ice-golem']
            elixir,threats=8.,()
        action=fast_policy.choose(hand,elixir,0.,[],features_from_threats(threats))
        print(json.dumps(dict(verified_templates=len(CARD_COST),hand=hand,elixir=elixir,
                              visible_threats=len(threats),card=action.card,wait=action.wait)),flush=True)
        return
    args.log.parent.mkdir(parents=True,exist_ok=True)
    with args.log.open('x',encoding='utf-8') as stream:
        def log(event,**data):
            print(json.dumps(dict(event=event,**data)),file=stream,flush=True)
        ready_event = 'ready_for_practice' if args.match_kind=='practice' else 'ready_for_normal_match'
        print(json.dumps({'event':ready_event,'mode':args.mode,**limits}),flush=True)
        deadline=time.monotonic()+95
        start=None
        gone=0
        pending=None
        last_defense=None
        own_events=[]
        latencies=[]
        defense_count=0
        capture_latencies=[]
        submitted=confirmed=failed=hog=missed=0
        result_image=args.log.with_suffix('.png')
        terminal_detected=False
        next_sample=0.
        if args.record_board:
            board_dir=args.log.with_name(args.log.stem+'-boards')
            board_dir.mkdir(exist_ok=True)
            def record_board(image,elapsed,kind):
                path=board_dir/f'{elapsed:07.2f}-{kind}.jpg'
                image.crop((50,200,850,1240)).resize((400,520)).save(path,quality=85)
        while time.monotonic()<deadline:
            if args.log.with_suffix('.STOP').exists():
                break
            cycle=time.monotonic()
            image=frame(adb)
            capture_latencies.append((time.monotonic()-cycle)*1000)
            live=battlefield(image)
            hand,quality=recognize(image) if live else ([],[])
            stable_hand=len(hand)==4 and None not in hand and len(set(hand))==4
            if start is None:
                if live and stable_hand:
                    start=time.monotonic()
                    deadline=start+args.seconds
                    if args.match_kind=='normal':
                        image.save(args.log.with_name(args.log.stem+'-opening.png'))
                    print(json.dumps({'event':'match_active','kind':args.match_kind,'mode':args.mode,'hand':hand}),flush=True)
                else:
                    time.sleep(.50)
                    continue
            elapsed=time.monotonic()-start
            if args.record_board and live and elapsed>=next_sample:
                record_board(image,elapsed,'board')
                next_sample=elapsed+4
            if not live:
                gone+=1
                if gone>=3:
                    image.save(result_image)
                    terminal_detected=True
                    break
            else:
                gone=0
            if pending and stable_hand:
                slot,card,target,requested,focus=pending
                if hand[slot]!=card:
                    x,y=target
                    own_events.append(dict(seconds=requested,side='team',event_type='card_play',card=card,x=x,y=y))
                    confirmed+=1
                    hog+=card=='hog-rider'
                    if focus is not None:
                        last_defense=(elapsed,*focus)
                        defense_count+=1
                    log('confirmed',card=card,slot=slot,seconds=round(elapsed,2))
                    pending=None
                elif elapsed-requested>4.5:
                    failed+=1
                    log('unconfirmed',card=card,slot=slot,seconds=round(elapsed,2))
                    pending=None
            elixir=elixir_value(image) if stable_hand else None
            if not stable_hand or elixir is None:
                missed+=1
            elif pending is None:
                begin=time.perf_counter()
                threats=visible_threats(image)
                action=None
                if args.mode=='selected' and not args.legacy_placement:
                    action=defense_move(hand,elixir,threats,last_defense,elapsed)
                if action is None:
                    action=fast_policy.choose(hand,elixir,elapsed,own_events,features_from_threats(threats))
                latency=(time.perf_counter()-begin)*1000
                latencies.append(latency)
                log('decision',elapsed=round(elapsed,2),hand=hand,confidence=quality,elixir=round(elixir,2),
                    wait=action.wait,card=action.card,reason=action.reason,
                    target=[action.x,action.y] if not action.wait else None,
                    visible_threats=[[t.x,t.y] for t in threats],inference_ms=round(latency,2))
                if not action.wait:
                    if args.record_board and action.focus is not None:
                        record_board(image,elapsed,action.card)
                    x,y=action.x,action.y
                    target_x=round(75+x*750/18000)
                    target_y=round(1200-y*1040/32000)
                    drag=round(min(520,260+.35*math.hypot(target_x-HAND_X[action.slot],target_y-HAND_Y)))
                    subprocess.run(adb+['shell','input','touchscreen','swipe',str(HAND_X[action.slot]),str(HAND_Y),
                                       str(target_x),str(target_y),str(drag)],check=True,timeout=5,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    pending=(action.slot,action.card,(x,y),elapsed,action.focus)
                    submitted+=1
            time.sleep(max(0.,1-(time.monotonic()-cycle)))
        if start is None:
            raise RuntimeError('No validated match hand appeared')
        summary=dict(mode=args.mode,match_kind=args.match_kind,observation='screen_only',terminal_detected=terminal_detected,
                     decisions=len(latencies),submitted=submitted,confirmed=confirmed,unconfirmed=failed,
                     hog_plays=hog,defensive_plays=defense_count,missing_frames=missed,memory=memory(),
                     elapsed_seconds=round(time.monotonic()-start,2),
                     median_inference_ms=round(statistics.median(latencies),2) if latencies else None,
                     median_capture_ms=round(statistics.median(capture_latencies),2),
                     result_image=str(result_image) if terminal_detected else None)
        log('complete',**summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
