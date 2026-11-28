"""Pair the deployed replay-based defense and the fitted Bush table on isolated pushes."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace as NS

from train_bush_defense import CARDS, NATIVE, run_trials, trials
from bush_defense import BushResponse
from runtime import configure
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tables','firstlight','reference','models','bush-table','report'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--classpath',required=True)
    args=parser.parse_args()
    configure(args.firstlight,args.reference)
    from support import limit_cpu
    limit_cpu()
    from defense_trials import TOWERS
    from defense_responses import ResponseTable, Unit
    from response_context import ContextHead
    from memory_policy import MemoryPolicy
    from live_decision import CARD_COST, DECK
    weights=BushResponse(args.bush_table)
    policy=object.__new__(MemoryPolicy)
    with np.load(args.models/'tile-prior.npz',allow_pickle=False) as prior:
        if tuple(prior['cards'])!=DECK:
            raise ValueError('Unexpected placement-prior card order')
        policy.prior=prior['log_probs'].reshape(8,3,576)
    policy.responses=ResponseTable(args.models/'defense-responses.npz')
    policy.context=ContextHead(args.models/'defense-context.npz')
    policy.match_strategy=policy.recent=policy.elixir_cap=None
    policy.allies=[]
    policy.builder=NS(pending=[])
    cases=trials(733103,8)
    prefixes=[' '.join(map(str,setup)) for setup,_,_ in cases]
    waits=run_trials([p+' wait 0 0' for p in prefixes],args.classpath,args.tables)
    rows=[]
    selected=[Counter(),Counter()]
    for case,wait,prefix in zip(cases,waits,prefixes):
        (x,y,count,pct,*_),hand,cash=case
        towers=[NS(**vars(t),hitpoints=wait[1],max_hitpoints=wait[1]) for t in TOWERS]
        lane_tower=next(t for t in towers if t.owner==0 and t.tower_kind==('princess_left' if x<9000 else 'princess_right'))
        lane_tower.hitpoints=int(wait[1]*pct/100)
        slots=[CARD_COST[c]<=cash for c in hand]
        masks={}
        for i,card in enumerate(hand):
            mask=np.zeros((32,18),dtype=bool)
            if slots[i]:
                mask[:32 if card in ('fireball','the-log') else 16]=True
                if card not in ('fireball','the-log'):
                    for t in towers:
                        if t.owner==0:
                            a,b=(int(v//1000) for v in t.position)
                            mask[b-1:b+2,a-1:a+2]=False
            masks[str(i)]={'row_major':mask}
        obs=NS(tick=400,towers=towers,players=[NS(hand=hand,elixir_exact=cash)],
               action_mask=NS(reasons={'effective_elixir':cash},hand_slots=slots,placement_masks=masks))
        policy.inputs=NS(card=lambda c:c,events=[dict(side='opponent',card='suspicious-bush',x=x,y=y,seconds=20)])
        for i,name in enumerate(('suspicious-bush','bush-goblin')):
            policy.bush_response=weights if i else None
            units=[Unit(j,26000097,name,x+(0 if count==1 else -500+j*1000),y,False,False,wait[2],first_y=y)
                   for j in range(count)]
            move,_=policy.plan(obs,units,None)
            card='wait' if move.wait else NATIVE[CARDS.index(move.card)]
            selected[i][move.card or 'wait']+=1
            rows.append(f'{prefix} {card} {move.x} {move.y}')
    hp=run_trials(rows,args.classpath,args.tables)[:,0].reshape(-1,2)
    summary=dict(pairs=len(cases),previous_survived=int((hp[:,0]>0).sum()),trained_survived=int((hp[:,1]>0).sum()),
                 previous_mean_hp=round(float(hp[:,0].mean()),1),trained_mean_hp=round(float(hp[:,1].mean()),1),
                 wins=int((hp[:,1]>hp[:,0]).sum()),ties=int((hp[:,1]==hp[:,0]).sum()),
                 losses=int((hp[:,1]<hp[:,0]).sum()),previous_choices=selected[0],trained_choices=selected[1])
    args.report.write_text(json.dumps(dict(summary=summary,paired_hp=hp.astype(int).tolist()),indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
