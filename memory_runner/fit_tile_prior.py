"""Fit a small card/ground-tank/lane deployment prior from disjoint IL_Replay splits."""
import argparse
import json
import os
from pathlib import Path
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

CARDS = ('hog-rider','musketeer','ice-golem','skeletons','ice-spirit','cannon','fireball','the-log')
TANKS = {'electro-giant','giant','golem','goblin-giant','royal-giant','elixir-golem'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.cache/'manifest.json').read_text())
    vocab=manifest['cards']
    own=np.full(max(vocab.values())+1,-1,dtype=np.int16)
    tank=np.zeros(len(own),dtype=bool)
    for name,index in vocab.items():
        name=re.sub(r'-(?:ev\d+|hero)$','',name)
        if name in CARDS:
            own[index]=CARDS.index(name)
        tank[index]=name in TANKS
    counts=np.zeros((3,len(CARDS),3,18,32),dtype=np.int64)
    matches=np.zeros(3,dtype=int)
    for path in sorted(args.cache.glob('*.npz')):
        with np.load(path) as shard:
            decks,splits=own[shard['decks']],shard['splits']
            near=((decks==0).any(2)&(decks==5).any(2)&((decks>=0).sum(2)>=6))
            if not near.any():
                continue
            data=shard['events']
            lengths=shard['counts']
            matches+=np.bincount(splits[near.any(1)],minlength=3)
            for side in (0,1):
                select=near[:,side]
                if not select.any():
                    continue
                events=data[select].astype(np.int32)
                ids,ticks,sides,x,y,slots,prefix=events.transpose(2,0,1)
                cols=np.arange(ids.shape[1])[None,:]
                valid=(cols<lengths[select,None])&(slots>=0)&(x>=0)
                last=np.maximum.accumulate(np.where(valid&(sides!=side)&tank[ids],cols,-1),axis=1)
                # Ignore all plays at the current tick, including simultaneous opposing plays.
                idx=np.take_along_axis(last,np.maximum(prefix-1,0),axis=1)
                known=(prefix>0)&(idx>=0)
                idx=np.maximum(idx,0)
                age=ticks-np.take_along_axis(ticks,idx,axis=1)
                nearby=known&(age>0)&(age<=14*20)
                context=np.where(nearby,1+(np.take_along_axis(x,idx,axis=1)>=9000),0)
                chosen=valid&(sides==side)&(own[ids]>=0)
                row=np.clip((32000-y if side else y)*18//32000,0,17)
                col=np.clip(x*32//18000,0,31)
                split=np.broadcast_to(splits[select,None],ids.shape)
                np.add.at(counts,tuple(a[chosen] for a in (split,own[ids],context,row,col)),1)
    train,_,test=counts
    # Mirror left/right tank encounters into the same geometry, then unmirror at lookup.
    pooled=train[:,1]+train[:,2,:,::-1]
    base=train.sum(1).astype(float)+.25
    base/=base.sum((1,2),keepdims=True)
    hist=np.stack((train[:,0],pooled,pooled[:,:,::-1]),axis=1)+100*base[:,None]
    probs=hist/hist.sum((2,3),keepdims=True)
    delta=np.log(probs/base[:,None])
    n=int(test.sum())
    tank_n=int(test[:,1:].sum())
    print(json.dumps(dict(train_matches=int(matches[0]),test_matches=int(matches[2]),train_placements=int(train.sum()),
                          test_placements=n,test_tank_placements=tank_n,
                          heldout_gain_nats_per_play=float((test*delta).sum()/n),
                          heldout_gain_after_tank=float((test[:,1:]*delta[:,1:]).sum()/max(tank_n,1)),
                          cards=list(CARDS)),indent=2))
    np.savez_compressed(args.output,log_probs=np.log(probs).astype('float32'),cards=np.asarray(CARDS))


if __name__=='__main__':
    main()
