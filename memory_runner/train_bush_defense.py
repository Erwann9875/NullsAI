"""Fit a small table from paired CRForge outcomes against revealed Bush Goblins."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from bush_defense import CARDS, COST, SHAPE, context, positions


NATIVE = ('Skeletons','IceSpirits','Log','IceGolemite','Musketeer','Fireball','Cannon')


def trials(seed, repeats):
    rng = random.Random(seed)
    cases = []
    for _ in range(repeats):
        for far in (False,True):
            for count in (1,2):
                for low in (False,True):
                    left = rng.choice((True,False))
                    edge = rng.randint(2800,4350)
                    x = edge if left else 18000-edge
                    y = rng.randint(11200,14300) if far else rng.randint(8650,10750)
                    hp = rng.randint(12,40) if low else rng.randint(49,100)
                    setup = (x,y,count,hp,rng.randint(20,34),rng.randint(0,15),rng.choice((11,14,16)))
                    hand = rng.sample([*CARDS,'hog-rider'],4)
                    cash = rng.choice((2.,3.,4.,5.,7.))
                    cases.append((setup,hand,cash))
    return cases


def run_trials(rows, classes, tables):
    command = ['java','-XX:ActiveProcessorCount=1','-XX:+UseSerialGC','-Xmx192m','-cp',str(classes),
               str(Path(__file__).with_name('BushDefenseTrials.java')),str(tables)]
    result = subprocess.run(command,input='\n'.join(rows)+'\n',capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    hp = np.array([[int(v) for v in row.split()] for row in result.stdout.splitlines()],dtype=float)
    if hp.shape != (len(rows),3) or np.any(hp[:,1:]<=0) or np.any(hp[:,0]>hp[:,1]):
        raise ValueError('Incomplete or invalid paired simulation results')
    return hp


def simulate(cases, classes, tables):
    rows = []
    for setup,_,_ in cases:
        x,y = setup[:2]
        prefix = ' '.join(map(str,setup))
        rows.append(prefix+' wait 0 0')
        for card,native in zip(CARDS,NATIVE):
            for a,b in positions(card,x,y,x<9000):
                rows.append(f'{prefix} {native} {a} {b}')
    return run_trials(rows,classes,tables).reshape(len(cases),1+2*len(CARDS),3)


def fit(cases, results):
    total = np.zeros(SHAPE)
    counts = np.zeros(SHAPE[:3])
    for (setup,_,_), hp in zip(cases,results):
        _,y,count,percent,*_ = setup
        key = context(y,count,percent/100)
        value = hp[:,0]/hp[:,1] + .35*(hp[:,0]>0)
        total[key] += (value[1:]-value[0]).reshape(len(CARDS),2) - .014*np.array(COST)[:,None]
        counts[key] += 1
    return (total/counts[...,None,None]).astype(np.float32)


def evaluate(cases, results, table):
    report = []
    for (setup,hand,elixir), hp in zip(cases,results):
        x,y,count,percent,*_ = setup
        scores = table[context(y,count,percent/100)]
        legal = [1+2*i+j for i,c in enumerate(CARDS) if c in hand and COST[i]<=elixir for j in (0,1)]
        selected = max([0,*legal],key=lambda a:0. if a==0 else scores.flat[a-1])
        cannon = 1+2*CARDS.index('cannon')
        baseline = cannon if cannon in legal else 0
        report.append(dict(hand=hand,elixir=elixir,setup=setup,
                           selected=CARDS[(selected-1)//2] if selected else 'wait',
                           cannon_available=bool(baseline),tower_hp=int(hp[selected,0]),
                           cannon_or_wait_hp=int(hp[baseline,0]),wait_hp=int(hp[0,0])))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classpath',required=True)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    train,test=trials(941303,10),trials(806291,8)
    results=simulate(train+test,args.classpath,args.tables)
    table=fit(train,results[:len(train)])
    rows=evaluate(test,results[len(train):],table)
    summary=dict(training_cases=len(train),held_out_cases=len(rows),
                 remaining_hp=sum(r['tower_hp'] for r in rows)/len(rows),
                 cannon_or_wait_hp=sum(r['cannon_or_wait_hp'] for r in rows)/len(rows),
                 towers_saved=sum(r['tower_hp']>0 for r in rows),
                 cannon_or_wait_survived=sum(r['cannon_or_wait_hp']>0 for r in rows),
                 choices={c:sum(r['selected']==c for r in rows) for c in (*CARDS,'wait')})
    args.report.write_text(json.dumps(dict(summary=summary,scenarios=rows),indent=2)+'\n',encoding='utf8')
    np.savez_compressed(args.out,cards=np.asarray(CARDS),scores=table)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
