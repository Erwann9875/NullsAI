"""Fit small card utilities and placement offsets from paired simulator rewards."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from defense_responses import CARDS, Unit, choose_counter
from defense_trials import OFFSETS


COST=dict(zip(CARDS,(4,4,2,1,1,3,4,2)),wait=0)


def read(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


def reward(result,cost):
    return result['hp']-650*result['lost']-.08*result['enemy_hp']-85*cost


def fit(rows,names):
    samples=defaultdict(list)
    for row in rows:
        wait=row['results'][0]
        for r in row['results'][1:]:
            if r['card'] in COST:
                samples[row['enemy'],r['card'],r['offset']].append(
                    reward(r,COST[r['card']])-reward(wait,0))
    gains=np.zeros((len(names),len(CARDS)),np.float32)
    offsets=np.zeros((*gains.shape,2),np.int8)
    for i,name in enumerate(names):
        for j,card in enumerate(CARDS):
            neutral=np.asarray(samples[name,card,0])
            if len(neutral)<5:
                continue
            best,offset=0.,0
            for k in range(1,len(OFFSETS)):
                results=samples[name,card,k]
                if not results:
                    continue
                delta=np.asarray(results)-neutral
                lower=delta.mean()-1.3*delta.std()/np.sqrt(len(delta))
                if lower>max(20,best):
                    best,offset=lower,k
            offsets[i,j]=OFFSETS[offset]
            shrink=len(neutral)/(len(neutral)+8)
            gains[i,j]=np.clip((neutral.mean()+best)*shrink/500,-4,5)
    return gains,offsets


def select(row,model,names,scale,use_offsets):
    u=Unit(**row['focus'])
    gain,offsets=model
    idx=names.index(u.name)
    results={(r['card'],r['offset']):r for r in row['results']}
    available=[c for c in CARDS if (c,0) in results]
    data=row['decision']
    scores=np.asarray(data['card_scores'] if data['card_scores'] is not None else np.zeros(8))
    group=[Unit(**t) for t in row['group'] if (t['x']-u.x)**2+(t['y']-u.y)**2<=2700**2]
    choice=choose_counter(available,u,group,scores+scale*gain[idx],data['spell_value'])
    if choice is None:
        return results['wait',0]
    off=tuple(offsets[idx,CARDS.index(choice)]) if use_offsets else (0,0)
    return results[choice,OFFSETS.index(off)]


def summary(rows,selected):
    return dict(cases=len(rows),mean_princess_hp=float(np.mean([r['princess'] for r in selected])),
                towers_lost=int(sum(r['lost'] for r in selected)),mean_elixir=float(np.mean(
                    [COST[row['baseline_card'] if r['card']=='baseline' else r['card']] for row,r in zip(rows,selected)])),
                reward=float(np.mean([reward(r,COST[row['baseline_card'] if r['card']=='baseline' else r['card']])
                                      for row,r in zip(rows,selected)])))


def compare(rows,model,names,scale,offsets):
    before=[next(r for r in row['results'] if r['card']=='baseline') for row in rows]
    current=[next(r for r in row['results'] if r['card']==row['policy_pick'] and r['offset']==0) for row in rows]
    after=[select(row,model,names,scale,offsets) for row in rows]
    delta=np.array([r['princess']-b['princess'] for r,b in zip(after,before)])
    # Resample whole groups of seeds, keeping the matchup mix fixed.
    groups=defaultdict(list)
    for row,d in zip(rows,delta):
        groups[row['seed']//100].append(d)
    means=np.array([np.mean(g) for g in groups.values()])
    rng=np.random.default_rng(91672)
    ci=np.quantile(rng.choice(means,(3000,len(means))).mean(1),[.025,.975]).tolist()
    return dict(previous=summary(rows,before),context_only=summary(rows,current),trained=summary(rows,after),
                paired_hp_gain=float(delta.mean()),paired_group_bootstrap_95=ci,
                by_threat={n:dict(hp_gain=float(delta[np.array([r['enemy']==n for r in rows])].mean()),
                                   cases=sum(r['enemy']==n for r in rows)) for n in sorted({r['enemy'] for r in rows})})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs='+',required=True,type=Path)
    for key in ('validation','table','out','report'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--test',type=Path)
    p.add_argument('--refine',nargs=3,action='append',metavar=('ENEMY','COUNTER','TRIAL_FILE'),
                   help='Refit only a specified matchup and counter from additional paired training trials.')
    a=p.parse_args()
    train=read(a.train)
    validation=read([a.validation])
    if {r['seed'] for r in train}&{r['seed'] for r in validation}:
        raise ValueError('Overlapping training and validation seeds')
    with np.load(a.table) as source:
        data={k:source[k] for k in source.files if k not in ('sim_offset','sim_gain')}
    names=list(data['enemies'])
    model=fit(train,names)
    settings=[]
    for scale in (0.,.5,1.,2.,4.,8.):
        for offsets in (False,True):
            selected=[select(r,model,names,scale,offsets) for r in validation]
            settings.append(dict(scale=scale,offsets=offsets,**summary(validation,selected)))
    best=max(settings,key=lambda s:s['reward'])
    scale,offsets=best['scale'],best['offsets']
    refinements=[]
    for enemy,card,path in a.refine or []:
        extra=[r for r in read([Path(path)]) if r['enemy']==enemy]
        if not extra or card not in CARDS or enemy not in names:
            raise ValueError('Unknown or empty counter refinement')
        if {r['seed'] for r in extra}&{r['seed'] for r in validation}:
            raise ValueError('Validation seeds present in counter refinement')
        gi,oi=fit(extra,names)
        i,j=names.index(enemy),CARDS.index(card)
        model[0][i,j],model[1][i,j]=gi[i,j],oi[i,j]
        train+=extra
        refinements.append(dict(enemy=enemy,card=card,training_cases=len(extra)))
    report=dict(training_cases=len(train),training_choices=sum(len(r['results']) for r in train),
                training_branches=sum(len({a['action'] for a in r['results']}) for r in train),
                refinements=refinements,
                validation=compare(validation,model,names,scale,offsets),validation_settings=settings,selected=best)
    if a.test:
        test=read([a.test])
        if {r['seed'] for r in test}&{r['seed'] for r in train+validation}:
            raise ValueError('Overlapping held-out seeds')
        report['test']=compare(test,model,names,scale,offsets)
    np.savez_compressed(a.out,**data,sim_gain=model[0]*scale,sim_offset=model[1]*int(offsets))
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k in ('training_cases','training_branches','selected')}))
    print(json.dumps({k:v for k,v in report.get('test',report['validation']).items() if k!='by_threat'}))


if __name__=='__main__':
    main()
