"""Legal candidate actions and public, lane-relative features shared by training and play."""
import numpy as np
from defense_responses import can_hit
from live_decision import CARD_COST, DECK, Move
from match_strategy import target_tower, tile_move
from rollout_features import canonical


def choices(obs,units,allies,ready,elixir):
    tower=target_tower(obs.towers,units)
    if not tower:
        return [Move()],np.zeros((1,64),np.float32)
    enemies=[u for u in units if can_hit(u)]
    lead=min(enemies,key=lambda u:u.y,default=None)
    second=min((u for u in enemies if lead and (lead.x<9000)!=(u.x<9000)),key=lambda u:u.y,default=None)
    enemy_broken=[t for t in obs.towers if t.owner==1 and t.tower_kind!='king' and not t.active]
    lane=3500 if tower.position[0]<9000 else 14500
    bridge=(lane,15500)
    attack=bridge
    if enemy_broken:
        attack=(6500 if enemy_broken[0].position[0]<9000 else 11500,21500)
    actions=[Move(reason='rollout_hold')]
    for card in ready:
        if card=='hog-rider':
            points=[bridge,(18000-lane,15500)]
            if enemy_broken:
                points.extend([attack,(lane,21500)])
        elif card=='fireball':
            points=[tower.position]+[(u.x,u.y-400) for u in sorted(enemies,key=lambda u:u.y)[:5]]
            if enemy_broken:
                points.extend(t.position for t in obs.towers if t.owner==1 and t.active)
        elif card=='the-log':
            points=[(lane,15500)]+[(u.x,max(500,u.y-3900)) for u in (lead,second) if u and not u.air]
        else:
            points=[attack,(lane,2500)]
            if enemy_broken:
                points.append(bridge)
            if card=='cannon':
                points=[(7500 if lane<9000 else 10500,10500)]
            for u in (lead,second):
                if not u: continue
                side=1 if u.x<9000 else -1
                near=3500 if side==1 else 14500
                if card=='cannon':
                    points.extend([(near+side*3000,10500),(near+side*1400,8500)])
                elif card=='musketeer':
                    points.extend([(near+side*3300,max(2500,min(10500,u.y-6000))),(near-side*1600,5500)])
                else:
                    points.extend([(u.x+1000*side,max(2500,min(14500,u.y-2700))),
                                   (near+side*6000,max(6500,min(13500,u.y-2300)))])
        unique=set()
        for point in points:
            move=tile_move(card,ready,obs,point,'rollout_'+card)
            if move and (move.x,move.y) not in unique:
                actions.append(move)
                unique.add((move.x,move.y))
    features=np.zeros((len(actions),64),dtype=np.float32)
    hp=[min(1,max(0,t.hitpoints/max(1,t.max_hitpoints))) for t in obs.towers]
    for i,m in enumerate(actions):
        f=features[i]
        f[0 if m.wait else 1+DECK.index(m.card)]=1.
        f[9:15]=hp
        f[15:20]=elixir/10, min(2.,obs.tick/3600),len(enemies)/8,len(allies)/6,float(bool(enemy_broken))
        x,y=m.x,m.y
        f[20:26]=x/18000,y/32000,(elixir-CARD_COST.get(m.card,0))/10,(x<9000)==(lane<9000),abs(x-9000)/9000,float(m.wait)
        for side in range(2):
            group=[u for u in enemies if ((u.x>=9000)==side)]
            a=min(group,key=lambda u:u.y,default=None)
            offset=26+side*9
            if a:
                f[offset:offset+9]=a.y/32000,float(a.buildings),float(bool(a.air)),len(group)/6,sum(bool(v.air) for v in group)/5,sum(v.name in ('prince','dark-prince') for v in group)/3,sum(v.buildings for v in group)/3,sum(v.name=='musketeer' for v in group)/3,min(1.,(a.hp or 0)/2500)
            else:
                f[offset]=1.
        for j,radius in enumerate((2500,5500,8500)):
            group=[u for u in enemies if (x-u.x)**2+(y-u.y)**2<radius**2]
            f[44+j*3:47+j*3]=len(group)/6,sum(bool(v.air) for v in group)/4,sum(v.buildings for v in group)/2
        f[53:57]=[sum(a.name==c and (a.x<9000)==(x<9000) for a in allies)/2 for c in ('cannon','musketeer','hog-rider','ice-golem')]
        f[57:61]=[c in ready for c in ('cannon','musketeer','hog-rider','fireball')]
        if lead:
            f[61:64]=abs(x-lead.x)/18000,(y-lead.y)/32000,(lead.x<9000)==(x<9000)
    return actions,features


class RolloutHead:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as weights:
            self.forest='roots' in weights
            self.canonicalize='canonical' in weights
            if self.forest:
                self.roots,self.left,self.right,self.indices,self.values,self.thresholds,self.depth=(
                    weights[k] for k in ('roots','left','right','features','values','thresholds','depth'))
            else:
                self.layers=[(weights['w'+str(i)],weights['b'+str(i)]) for i in range(3)]

    def scores(self,features):
        if self.canonicalize:
            features=canonical(features)
        if self.forest:
            nodes=np.broadcast_to(self.roots,(len(features),len(self.roots)))
            rows=np.arange(len(features))[:,None]
            for _ in range(int(self.depth)):
                nodes=np.where(features[rows,self.indices[nodes]]<=self.thresholds[nodes],self.left[nodes],self.right[nodes])
            return self.values[nodes].mean(1)
        for w,b in self.layers[:-1]:
            features=np.maximum(0,features@w+b)
        w,b=self.layers[-1]
        return (features@w+b).ravel()
