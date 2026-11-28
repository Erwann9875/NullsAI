"""Small replay-trained response head. Live inference uses one NumPy matrix per layer."""
import numpy as np

from fit_tile_prior import CARDS


def geometry(x,y,age,mirror):
    return ((np.where(mirror,17999-x,x)-9000)/9000, (y-16000)/16000,
            np.minimum(800,np.maximum(0,age))/800)


class ContextHead:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.names={str(name):i for i,name in enumerate(data['names'])}
            self.weights={k:data[k] for k in ('embedding','trunk','bias','choice','choice_bias',
                                                'tile_card','tile_hidden','tile_bias','tile','out_bias')}
            if tuple(data['cards'])!=CARDS or self.weights['embedding'].shape[0]!=len(self.names):
                raise ValueError('Incompatible defense context model')

    def encode(self,focus,events,hand,tick):
        ours,theirs=[],[]
        source=None
        for event in reversed(events):
            age=tick-round(event['seconds']*20)
            if age<0:
                continue
            entry=(event['card'],event['x'],event['y'],age)
            if event['side']=='team':
                if len(ours)<3:
                    ours.append(entry)
            else:
                if len(theirs)<3:
                    theirs.append(entry)
                if source is None and event['card']==focus.name and (event['x']<9000)==(focus.x<9000):
                    source=entry
            if len(ours)==3 and len(theirs)==3 and source is not None:
                break
        empty=('',0,0,0)
        entries=[source or (focus.name,focus.x,focus.first_y or focus.y,0),
                 *(ours+[empty]*3)[:3],*(theirs+[empty]*3)[:3]]
        ids=[self.names.get(e[0],0) for e in entries]
        nums=[]
        for name,x,y,age in entries:
            nums.extend(geometry(x,y,age,focus.x>=9000) if name else (0.,0.,0.))
        nums.extend([min(tick/7200,1.),*(float(c in hand) for c in CARDS)])
        return np.asarray(ids),np.asarray(nums,dtype='float32')

    def predict(self,focus,events,hand,tick):
        ids,nums=self.encode(focus,events,hand,tick)
        w=self.weights
        inputs=np.concatenate((w['embedding'][ids].ravel(),nums))
        hidden=np.maximum(0,inputs@w['trunk'].T+w['bias'])
        return hidden@w['choice'].T+w['choice_bias'],hidden

    def tiles(self,hidden,card,right):
        w=self.weights
        state=np.maximum(0,hidden@w['tile_hidden'].T+w['tile_bias']+w['tile_card'][CARDS.index(card)])
        scores=state@w['tile'].T+w['out_bias']
        ys,xs,zone=scores[:32],scores[32:50],scores[50:].reshape(8,6)
        surface=ys[:,None]+xs[None,:]+zone.repeat(4,0).repeat(3,1)
        return surface[:,::-1] if right else surface
