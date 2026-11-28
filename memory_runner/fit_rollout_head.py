"""Distill simulator action returns into a small single-thread CPU value head."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from support import limit_cpu


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('datasets',nargs='+',type=Path)
    cli.add_argument('--epochs',type=int,default=32)
    cli.add_argument('--forest',action='store_true')
    cli.add_argument('--ranking',action='store_true',help='Scale each board by its action-value spread and reinforce its best legal plays.')
    cli.add_argument('--canonical',action='store_true')
    cli.add_argument('--pairwise',action='store_true',help='Learn to rank the alternatives available on the same board.')
    a=cli.parse_args()
    limit_cpu()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(8319)
    cases=[]
    for path in a.datasets:
        with path.open('rb') as f:
            while f.peek(1):
                cases.append(np.load(f,allow_pickle=False))
    if a.ranking:
        for case in cases:
            y=case[:,-1]
            y/=max(.18,float(np.ptp(y)))
            y+=.25*(y>=y.max()-.03)
            y-=y[0]
    rows=torch.from_numpy(np.concatenate(cases))
    x,y=rows[:,:-1],rows[:,-1]
    marker={}
    if a.canonical:
        from rollout_features import canonical
        x=torch.from_numpy(canonical(x.numpy()))
        marker['canonical']=True
    if a.forest:
        from sklearn.ensemble import ExtraTreesRegressor
        forest=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=4,max_depth=16,max_features=.85,
                                   random_state=8319,n_jobs=1)
        importance=np.concatenate([np.full(len(c),.25+min(2.,np.ptp(c[:,-1]))/len(c)**.35) for c in cases])
        forest.fit(x.numpy(),y.numpy(),sample_weight=importance)
        offsets=[];left=[];right=[];values=[];features=[];thresholds=[]
        count=0
        for t in forest.estimators_:
            tree=t.tree_
            offsets.append(count)
            idx=np.arange(tree.node_count)
            left.extend(count+np.where(tree.children_left<0,idx,tree.children_left))
            right.extend(count+np.where(tree.children_right<0,idx,tree.children_right))
            values.extend(tree.value.ravel())
            features.extend(np.maximum(tree.feature,0))
            thresholds.extend(tree.threshold)
            count+=tree.node_count
        np.savez_compressed(a.output,roots=np.int32(offsets),left=np.int32(left),right=np.int32(right),
                            features=np.uint8(features),values=np.float32(values),thresholds=np.float32(thresholds),
                            depth=np.int32(max(t.tree_.max_depth for t in forest.estimators_)),**marker)
        print('Fitted',len(cases),'decisions,',len(x),'legal action returns.')
        return
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,48),nn.ReLU(),nn.Linear(48,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    best=torch.from_numpy(np.repeat(np.cumsum([0]+[len(c) for c in cases[:-1]])+
                                    np.array([c[:,-1].argmax() for c in cases]),[len(c) for c in cases]))
    for epoch in range(a.epochs):
        for batch in torch.randperm(len(x)).split(256):
            pred=model(x[batch]).squeeze(-1)
            loss=nn.functional.smooth_l1_loss(pred,y[batch],beta=.35)
            if a.pairwise:
                rivals=best[batch]
                delta=torch.clamp(y[rivals]-y[batch],0,1)
                advantage=model(x[rivals]).squeeze(-1)-pred
                loss=loss+.7*torch.relu(delta-advantage).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    weights={}
    for i,layer in enumerate(model[::2]):
        weights['w'+str(i)]=layer.weight.detach().numpy().T.copy()
        weights['b'+str(i)]=layer.bias.detach().numpy()
    np.savez_compressed(a.output,**weights,**marker)
    print('Fitted',len(cases),'decisions,',len(x),'legal action returns.')


if __name__=='__main__':
    main()
