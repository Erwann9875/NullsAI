"""Training labels from alternative legal actions on the same CRForge board."""
import numpy as np

from rollout_choices import choices


class RolloutTeacher:
    def __init__(self,stream,horizon,interval,objective='balanced'):
        self.stream=stream
        self.horizon=horizon
        self.interval=interval
        self.next_tick=0
        self.initial=objective=='initial'

    def choose(self,player,obs,units,allies,hand,elixir):
        from live_decision import Move
        if obs.tick<self.next_tick or player.pending:
            return Move(reason='teacher_wait'),None
        ready={c:i for i,c in enumerate(hand) if obs.action_mask.hand_slots[i]}
        if not ready:
            return Move(reason='teacher_wait'),None
        self.next_tick=obs.tick+self.interval
        anchors=[a for a in allies if a.name in {'cannon','musketeer','hog-rider','ice-golem'}]
        actions,features=choices(obs,units,anchors,ready,elixir)
        arena=player.arena
        s=arena.schema
        codes=np.array([0 if m.wait else s.play(m.slot,m.x//1000,m.y//1000) for m in actions])
        views,infos=arena.lookahead(np.full(len(codes),player.i),np.full(len(codes),player.seat),codes,self.horizon)
        if np.any(infos[:,s.info['fault']]):
            raise RuntimeError('Counterfactual rollout fault')
        h=lambda k: views[:,s.header[k][0]].astype(float)
        def towers(side):
            left,right,king=[np.maximum(0,h(side+'_'+part+'_hp')) for part in ('left','right','king')]
            return .55*(left+right)+.8*np.minimum(left,right)+1.35*king
        values=1.35*towers('own')-towers('enemy')+1200*(h('crowns')-h('crowns_enemy'))+155*(h('elixir')-h('elixir_reserved'))/10000
        if self.initial:
            own=h('own_left_hp')+h('own_right_hp')+1.3*h('own_king_hp')
            enemy=h('enemy_left_hp')+h('enemy_right_hp')+1.1*h('enemy_king_hp')
            values=1.25*own-enemy+750*(h('crowns')-h('crowns_enemy'))+105*(h('elixir')-h('elixir_reserved'))/10000
        for i,view in enumerate(views):
            entities=s.entities(view)
            active=entities[(entities[:,s.entity['kind']]==s.kinds['character']) & ((entities[:,s.entity['flags']]&s.flags['alive'])!=0)]
            for e in active:
                side=int(e[s.entity['side']])
                hp=int(e[s.entity['hp']])
                mx=max(1,int(e[s.entity['max_hp']]))
                if side not in (0,1): continue
                name=player.name(int(e[s.entity['row']]))
                cost=player.facts.get(name,{}).get('elixir',1)
                if name in {'Skeleton','Minion','skeletons','minions'}: cost=1
                relative=1 if hp<0 else min(1,hp/mx)
                mass=100*cost*(.3+.7*relative) if self.initial else 140*cost*(.25+.75*relative)
                values[i]+=(1 if side==0 else -1)*mass
        advantage=np.clip((values-values[0])/1000,-7,7).astype(np.float32)
        if self.stream:
            np.save(self.stream,np.column_stack((features,advantage)),allow_pickle=False)
        move=actions[int(advantage.argmax())]
        return move,None
