"""Bounded CRForge previews reconstructed from visible units, own hand and public tower HP."""
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from live_decision import CARD_COST


class PublicForecast:
    def __init__(self,args):
        sys.path.insert(0,str(args.simulator.resolve()/'python'))
        import jpype
        from crforge_env import _jvm
        java=shutil.which('java')
        if java:
            os.environ.setdefault('JAVA_HOME',str(Path(java).resolve().parents[1]))
        _jvm.start(args.simulator/'env/build/install/env',('-Xmx256m','-XX:ActiveProcessorCount=1','-XX:+UseSerialGC'))
        jpype.addClassPath(str(args.public_classes.resolve()))
        self.engine=jpype.JClass('PublicBoard')(str(args.tables.resolve()))
        catalog=json.loads((args.reference/'engine/data/cards.json').read_text(encoding='utf8'))
        self.rows={r['id'].replace('_','-'):r['sc_name'] for r in catalog}
        for r in catalog:
            self.engine.setCost(r['sc_name'],int(r.get('elixir') or 1))
        self.rows.update({'Minion':'Minion','Skeleton':'Skeleton'})
        self.rows['bush-goblin']='BushGoblin'
        self.costs=CARD_COST
        self.shadow=None
        self.seen=set()
        self.error=None
        if getattr(args,'player_tag',None):
            preview=self.engine.shadow()
            for tick in (60,80,100):
                preview.predict(tick,[1.]*6,['Giant'],[1,3500,15500,3000],['','Cannon','Musketeer'],
                                [0,0,0,6500,10500,3,5500,6500,4],100,6.,2.,155.,1.35,True,True)

    def reset(self):
        self.shadow=None
        self.seen=set()
        self.error=None

    def score(self,obs,units,allies,moves,student,strategy,elixir,events):
        import jpype
        if self.error:
            return student
        try:
            return self._score(obs,units,allies,moves,student,strategy,elixir,events)
        except jpype.JClass('java.lang.RuntimeException') as failure:
            self.error=str(failure)
            return student

    def _score(self,obs,units,allies,moves,student,strategy,elixir,events):
        indices=np.arange(len(moves))
        if len(indices)>strategy.search_width:
            ranking=(1+np.argsort(-student[1:],kind='stable')).tolist()
            if strategy.diverse:
                first={}
                for idx in ranking:
                    first.setdefault(moves[idx].card,idx)
                priority=list(first.values())
                broken=any(t.owner==1 and t.tower_kind!='king' and not t.active for t in obs.towers)
                if broken:
                    hog=[idx for idx,m in enumerate(moves) if m.card=='hog-rider']
                    if hog:
                        priority.append(hog[0])
                        priority.extend(idx for idx in hog if moves[idx].y>=18500)
                ranking=list(dict.fromkeys([*priority,*ranking]))
            indices=np.array([0,*ranking[:strategy.search_width-1]])
        chosen=[moves[i] for i in indices]
        scale=3052/max(1,next(t.max_hitpoints for t in obs.towers if t.owner==0 and t.tower_kind!='king'))
        state=[];names=[]
        for side,group in enumerate((allies,units)):
            for u in group:
                name=self.rows.get(u.name,u.name)
                names.append(name)
                state.extend((side,int(u.x),int(u.y),-1 if u.hp is None else round(u.hp*scale)))
        towers=[0.]*6
        for t in obs.towers:
            towers[t.owner*3+('king','princess_left','princess_right').index(t.tower_kind)]=max(0,t.hitpoints/max(1,t.max_hitpoints))
        cards=[self.rows[m.card] if m.card else '' for m in chosen]
        plays=[v for m in chosen for v in (m.x,m.y,self.costs.get(m.card,0))]
        regeneration=strategy.horizon*.05/2.8*(2 if obs.tick>=2400 else 1)
        if self.shadow is None:
            self.shadow=self.engine.shadow()
        for event in events:
            name=self.rows.get(event['card'])
            if not name: continue
            key=(round(event['seconds']*20),name,int(event['side']!='team'),int(event['x']),int(event['y']))
            if key not in self.seen:
                self.seen.add(key)
                self.shadow.add(*key)
        result=np.array(self.shadow.predict(obs.tick,towers,names,state,cards,plays,strategy.horizon,float(elixir),regeneration,
                                            strategy.reserve_value,strategy.defense_value,strategy.correct_board,strategy.by_cost))
        scores=np.full(len(moves),-np.inf)
        scores[indices]=(result-result[0])/1000+strategy.student_value*student[indices]
        return scores
