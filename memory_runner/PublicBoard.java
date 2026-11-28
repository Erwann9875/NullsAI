import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.ArrayList;
import java.util.TreeMap;
import org.crforge.core.util.DeepCopy;
import org.crforge.core.pathfinding.combat.RarityTable;
import org.crforge.core.battle.BattleEntity;
import org.crforge.core.battle.data.GameTables;
import org.crforge.core.battle.data.BattleRecords;
import org.crforge.core.battle.deploy.DeployCard;
import org.crforge.core.battle.unit.CharacterEntity;
import org.crforge.core.battle.unit.Standard1v1Battle;
import org.crforge.core.battle.unit.TowerEntity;
import org.crforge.core.battle.unit.UnitData;
import org.crforge.core.battle.unit.WorldEntity;

/** Forecasts only the visible board supplied by the caller. No game client or opponent hand is used. */
public final class PublicBoard {
    private final GameTables tables;
    private final BattleRecords records;
    private final Map<String, UnitData> units = new HashMap<>();
    private final Map<String, DeployCard> cards = new HashMap<>();
    private final Map<String,Integer> costs=new HashMap<>();
    private final Set<Object> fixed;

    public PublicBoard(String folder) {
        tables=GameTables.load(Path.of(folder));
        records=new BattleRecords(tables);
        fixed=DeepCopy.reachable(tables);
    }

    public void setCost(String unit,int cost) { costs.put(unit,cost); }

    public final class Shadow {
        private record Event(String name,int side,int x,int y) {}
        private final TreeMap<Integer,ArrayList<Event>> tape=new TreeMap<>();
        private Standard1v1Battle base=new Standard1v1Battle(tables,11);
        private DeepCopy.Memo memo=new DeepCopy.Memo();
        private int clock=0;

        public void add(int tick,String name,int side,int x,int y) {
            tick=Math.max(20,tick);
            Event event=new Event(name,side,x,y);
            ArrayList<Event> at=tape.computeIfAbsent(tick,k->new ArrayList<>());
            if (at.contains(event)) return;
            at.add(event);
            if (tick<clock) {
                base=new Standard1v1Battle(tables,11);
                memo=new DeepCopy.Memo();
                clock=0;
            }
        }

        private Standard1v1Battle fork(Standard1v1Battle b) {
            return DeepCopy.of(b,o->o instanceof RarityTable || o==records || fixed.contains(o),memo);
        }

        private void advance(Standard1v1Battle b,int start,int end) {
            for (var pair:tape.subMap(start,true,end,false).entrySet()) {
                int i=0;
                for (Event e:pair.getValue()) {
                    DeployCard card=cards.computeIfAbsent(e.name(),records::card);
                    b.play(pair.getKey(),card,11,e.side(),e.x(),e.y(),"p"+pair.getKey()+"_"+i++);
                }
            }
            for (int tick=start;tick<end;tick++) b.getBattle().step();
        }

        public double[] predict(int now,double[] towers,String[] observed,int[] state,String[] moves,int[] plays,int horizon,
                                double elixir,double generation,double reserve,double defense,boolean correct,boolean byCost) {
            int stable=Math.max(0,now-60);
            if (stable>clock) {
                advance(base,clock,stable);
                clock=stable;
            }
            Standard1v1Battle current=fork(base);
            advance(current,clock,now);
            for (BattleEntity entity:current.getBattle().getHolder().entities()) {
                if (entity instanceof TowerEntity t) {
                    int slot=3*t.side()+(t.getData().king()?0:t.x()<9000?1:2);
                    t.getHitPoints().setHitPoints((int)Math.round(t.getHitPoints().getMaximum()*towers[slot]));
                }
            }
            if (correct) synchronize(current,observed,state,now);
            double[] scores=new double[moves.length];
            for(int k=0;k<moves.length;k++) {
                Standard1v1Battle branch=fork(current);
                if (!moves[k].isEmpty()) {
                    branch.play(now+20,cards.computeIfAbsent(moves[k],records::card),11,0,plays[3*k],plays[3*k+1],"proposal");
                }
                for(int t=0;t<horizon;t++) branch.getBattle().step();
                scores[k]=evaluate(branch,elixir-plays[3*k+2]+generation,reserve,defense,byCost,towers);
            }
            return scores;
        }
    }

    public Shadow shadow() { return new Shadow(); }

    private void synchronize(Standard1v1Battle board,String[] names,int[] state,int now) {
        ArrayList<CharacterEntity> remaining=new ArrayList<>();
        for (BattleEntity entity:board.getBattle().getHolder().entities()) {
            if (entity instanceof CharacterEntity c && c.getHitPoints()!=null) remaining.add(c);
        }
        for (int i=0;i<names.length;i++) {
            UnitData type=unit(names[i]);
            if(type==null || type.spawnAttach()) continue;
            int j=4*i,side=state[j],x=state[j+1],y=state[j+2],hp=state[j+3];
            CharacterEntity chosen=null;
            long nearest=Long.MAX_VALUE;
            for (CharacterEntity c:remaining) {
                if(c.side()!=side || !c.getData().name().equals(type.name())) continue;
                long dx=c.x()-x,dy=c.y()-y,d=dx*dx+dy*dy;
                if(d<nearest) { nearest=d; chosen=c; }
            }
            if(chosen==null) {
                chosen=board.deploy(now,type,11,side,x,y,"observed"+i);
                chosen.getView().setDeployCountdown(0);
            } else {
                remaining.remove(chosen);
                if(nearest>500*500) {
                    chosen.getView().setX(x);
                    chosen.getView().setY(y);
                }
            }
            if(hp>=0) chosen.getHitPoints().setHitPoints(Math.min(chosen.getHitPoints().getMaximum(),Math.max(1,hp)));
        }
        for(CharacterEntity c:remaining) c.getHitPoints().setHitPoints(0);
    }

    private double evaluate(Standard1v1Battle board,double balance,double reserve,double defense,boolean byCost,double[] observed) {
        double[] hp=new double[6];
        double score=reserve*Math.min(10,balance);
        for (BattleEntity entity:board.getBattle().getHolder().entities()) {
            if (!(entity instanceof WorldEntity u) || u.getHitPoints()==null) continue;
            int health=Math.max(0,u.getHitPoints().getHitPoints());
            if (u instanceof TowerEntity tower) {
                hp[3*u.side()+(tower.getData().king()?0:u.x()<9000?1:2)]=health;
            } else if (u instanceof CharacterEntity && health>0) {
                int cost=costs.getOrDefault(u.getData().name(),1);
                double weight=byCost?140*cost*(.25+.75*health/Math.max(1.,u.getHitPoints().getMaximum())):.18*health+55;
                score+=u.side()==0?weight:-weight;
            }
        }
        for(int s=0;s<2;s++){
            int j=3*s;double l=hp[j+1],r=hp[j+2],k=hp[j];
            double v=.55*(l+r)+.8*Math.min(l,r)+1.35*k+1200*((l>0?1:0)+(r>0?1:0))+5500*(k>0?1:0);
            if ((observed[j+1]<=0)!=(observed[j+2]<=0)) {
                double standing=observed[j+1]>0?l:r;
                double originalPrincess=3052*Math.max(observed[j+1],observed[j+2]);
                if(originalPrincess<=4824*observed[j]) v+=1.10*standing-.7*k;
            }
            score+=s==0?defense*v:-v;
        }
        return score;
    }

    private UnitData unit(String name) {
        if (!units.containsKey(name)) {
            UnitData found;
            try { found=records.unit(name); }
            catch (RuntimeException e) {
                try { found=records.card(name).unit(); }
                catch (RuntimeException ignored) { found=null; }
            }
            units.put(name,found);
        }
        return units.get(name);
    }
}
