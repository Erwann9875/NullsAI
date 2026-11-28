"""Train a compact hand- and history-conditioned head on a reusable causal replay cache."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fit_tile_prior import CARDS
from fit_defense_responses import family
from response_context import geometry


def build(cache,catalog,path):
    manifest=json.loads((cache/'manifest.json').read_text())
    vocab=manifest['cards']
    names=['']+sorted({family(n) for n in vocab if n not in ('<pad>','_unknown_ability')})
    ids=np.zeros(max(vocab.values())+1,dtype=np.int16)
    own=np.full(len(ids),-1,dtype=np.int16)
    threat=np.zeros(len(ids),dtype=bool)
    enemies={r['id'].replace('_','-') for r in catalog if r['kind']=='troop'}|{'x-bow','mortar','goblin-barrel'}
    for name,i in vocab.items():
        n=family(name)
        ids[i]=names.index(n) if n in names else 0
        threat[i]=n in enemies
        if n in CARDS:
            own[i]=CARDS.index(n)
    parts=[]
    for shard in sorted(cache.glob('*.npz')):
        with np.load(shard) as data:
            decks=own[data['decks']]
            hog=(decks==0).any(2)&(decks==5).any(2)&((decks>=0).sum(2)>=6)
            all_events=data['events']
            for side in (0,1):
                subset=hog[:,side]
                if not subset.any():
                    continue
                a,t,s,x,y,slots,prefix=all_events[subset].astype(np.int32).transpose(2,0,1)
                if side:
                    y=31999-y
                index=np.arange(a.shape[1])[None,:]
                valid=(index<data['counts'][subset,None])&(slots>=0)&(x>=0)
                def previous(mask):
                    last=np.maximum.accumulate(np.where(mask,index,-1),axis=1)
                    return np.where(prefix>0,np.take_along_axis(last,np.maximum(0,prefix-1),axis=1),-1)
                get=lambda values,idx:np.take_along_axis(values,np.maximum(0,idx),axis=1)
                focus=previous(valid&(side!=s)&threat[a])
                age=t-get(t,focus)
                pick=valid&(side==s)&(own[a]>=0)&(focus>=0)&(age>0)&(age<=160)
                if not pick.any():
                    continue
                earlier=[focus]
                own_history=[]
                for team in (side,1-side):
                    prev=previous(valid&(s==team))
                    idx=prev
                    for level in range(4):
                        if level<3:
                            earlier.append(idx)
                        if team==side:
                            own_history.append(idx)
                        idx=np.where(idx>=0,get(prev,idx),-1)
                mirror=get(x,focus)[pick]>=9000
                cats=[]
                floats=[]
                for idx in earlier:
                    known=idx[pick]>=0
                    cats.append(np.where(known,get(ids[a],idx)[pick],0))
                    for v in geometry(get(x,idx)[pick],get(y,idx)[pick],(t-get(t,idx))[pick],mirror):
                        floats.append(np.where(known,v,0.))
                floats.append(np.minimum(t[pick]/7200,1.))
                deck=decks[subset,side]
                for card in range(len(CARDS)):
                    available=np.broadcast_to((deck==card).any(1)[:,None],a.shape).copy()
                    for idx in own_history:
                        available &= (idx<0)|(get(own[a],idx)!=card)
                    floats.append(available[pick])
                split=np.broadcast_to(data['splits'][subset,None],a.shape)[pick]
                labels=np.stack([own[a][pick],y[pick]//1000,np.where(mirror,17-x[pick]//1000,x[pick]//1000),split],1)
                parts.append((np.stack(cats,1).astype('int16'),np.stack(floats,1).astype('float16'),labels.astype('uint8')))
    cats,features,labels=[np.concatenate(column) for column in zip(*parts)]
    np.savez(path,cats=cats,features=features,labels=labels,names=names,revision=manifest['dataset_revision'])


class Head(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.embedding=nn.Embedding(n,8,padding_idx=0)
        self.trunk=nn.Linear(7*8+30,64)
        self.choice=nn.Linear(64,8)
        self.tile_card=nn.Embedding(8,48)
        self.tile_hidden=nn.Linear(64,48)
        self.tile=nn.Linear(48,98)

    def forward(self,cat,f,card):
        h=F.relu(self.trunk(torch.cat((self.embedding(cat).flatten(1),f),1)))
        position=self.tile(F.relu(self.tile_hidden(h)+self.tile_card(card)))
        return self.choice(h),position.split((32,18,48),1)


def evaluate(net,indices,data,table,names):
    cats,features,labels=data
    name_map={str(n):i for i,n in enumerate(table['enemies'])}
    lookup=np.array([name_map.get(str(n),0) for n in names])
    total=np.zeros(9)
    with torch.inference_mode():
        for ix in indices.split(2048):
            cs=cats[ix].long()
            f=features[ix].float()
            card,y,x=labels[ix,:3].long().T
            mask=f[:,-8:]>0
            scores,(ys,xs,zones)=net(cs,f,card)
            enemy=lookup[cs[:,0].numpy()]
            depth=(f[:,1].numpy()>-3000/16000).astype(int)+(f[:,1].numpy()>5000/16000)
            prior=torch.from_numpy(table['card_logp'][enemy,depth])
            legal=mask[torch.arange(len(ix)),card]
            tiles=ys[:,:,None]+xs[:,None,:]+zones.reshape(-1,8,6).repeat_interleave(4,1).repeat_interleave(3,2)
            best=tiles.flatten(1).argmax(1)
            table_tiles=table['tile_logp'][enemy,depth,card.numpy()].reshape(len(ix),-1)
            ref=np.argmax(table_tiles,1)
            total += [len(ix),legal.sum().item(),(scores.masked_fill(~mask,-1e6).argmax(1)==card).sum().item(),
                      (prior.masked_fill(~mask,-1e6).argmax(1)==card).sum().item(),
                      (scores.argmax(1)==card).sum().item(),(prior.argmax(1)==card).sum().item(),
                      torch.hypot((best//18-y).float(),(best%18-x).float()).sum().item(),
                      np.hypot(ref//18-y.numpy(),ref%18-x.numpy()).sum(),(best==y*18+x).sum().item()]
    return dict(zip(('examples','label_in_hand','card_accuracy','table_card_accuracy','unmasked_card_accuracy',
                     'table_unmasked_card_accuracy','tile_error','table_tile_error','exact_tile'),
                    [int(total[0]),*(total[1:]/total[0])]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','catalog','examples','table','out','report'):
        p.add_argument('--'+arg,required=True,type=Path)
    p.add_argument('--epochs',type=int,default=14)
    args=p.parse_args()
    from support import limit_cpu
    limit_cpu()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(19713)
    start=time.monotonic()
    if not args.examples.exists():
        build(args.cache,json.loads(args.catalog.read_text(encoding='utf8')),args.examples)
    with np.load(args.examples,allow_pickle=False) as data:
        revision=json.loads((args.cache/'manifest.json').read_text())['dataset_revision']
        if str(data['revision'])!=revision:
            raise ValueError('The cached response examples belong to another dataset revision')
        cats,features,labels=[torch.from_numpy(data[k]) for k in ('cats','features','labels')]
        names=data['names']
    cache_s=time.monotonic()-start
    indices=[torch.where(labels[:,3]==s)[0] for s in range(3)]
    with np.load(args.table) as source:
        table={k:source[k] for k in ('enemies','card_logp','tile_logp')}
    model=Head(len(names))
    opt=torch.optim.AdamW(model.parameters(),lr=.0025)
    history=[]
    best=float('inf')
    best_epoch=0
    for epoch in range(1,args.epochs+1):
        es=time.monotonic()
        order=indices[0][torch.randperm(len(indices[0]))]
        for ix in order.split(1024):
            f=features[ix].float()
            card,y,x=labels[ix,:3].long().T
            scores,(ys,xs,zones)=model(cats[ix].long(),f,card)
            mask=f[:,-8:]>0
            # Broken/special rotations have ambiguous hands. Keep the unmasked training signal.
            legal=mask[torch.arange(len(ix)),card]
            masked=scores.masked_fill(~mask&legal[:,None],-1e4)
            loss=F.cross_entropy(masked,card)+.2*F.cross_entropy(scores,card)
            loss+=.55*(F.cross_entropy(ys,y)+F.cross_entropy(xs,x))+.45*F.cross_entropy(zones,(y//4)*6+x//3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m=evaluate(model,indices[1],(cats,features,labels),table,names)
        criterion=m['tile_error']-8*m['card_accuracy']
        history.append(dict(epoch=epoch,seconds=time.monotonic()-es,**m))
        print(json.dumps(history[-1]),flush=True)
        if criterion<best:
            best,best_epoch=criterion,epoch
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=3:
            break
    model.load_state_dict(state)
    test=evaluate(model,indices[2],(cats,features,labels),table,names)
    keys={'embedding':'embedding.weight','trunk':'trunk.weight','bias':'trunk.bias','choice':'choice.weight',
          'choice_bias':'choice.bias','tile_card':'tile_card.weight','tile_hidden':'tile_hidden.weight',
          'tile_bias':'tile_hidden.bias','tile':'tile.weight','out_bias':'tile.bias'}
    np.savez_compressed(args.out,names=names,cards=CARDS,**{k:state[v].numpy() for k,v in keys.items()})
    report=dict(examples=[len(v) for v in indices],cache_seconds=cache_s,elapsed_seconds=time.monotonic()-start,
                selected_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),threads=1,
                history=history,test=test)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(test),flush=True)


if __name__=='__main__':
    main()
