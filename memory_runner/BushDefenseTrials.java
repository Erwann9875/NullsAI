import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import org.crforge.core.battle.BattleEntity;
import org.crforge.core.battle.data.BattleRecords;
import org.crforge.core.battle.data.GameTables;
import org.crforge.core.battle.unit.CharacterEntity;
import org.crforge.core.battle.unit.Standard1v1Battle;
import org.crforge.core.battle.unit.TowerEntity;

/** Each row runs one action against the same visible goblins as its paired wait. */
public final class BushDefenseTrials {
    private static TowerEntity princess(Standard1v1Battle arena, boolean left) {
        for (BattleEntity entity : arena.getBattle().getHolder().entities()) {
            if (entity instanceof TowerEntity tower && tower.side() == 0
                    && !tower.getData().king() && (tower.x() < 9000) == left) return tower;
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        GameTables tables = GameTables.load(Path.of(args[0]));
        BattleRecords records = new BattleRecords(tables);
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = input.readLine()) != null) {
            String[] f = line.split(" ");
            int x = Integer.parseInt(f[0]), y = Integer.parseInt(f[1]);
            int count = Integer.parseInt(f[2]), healthPercent = Integer.parseInt(f[3]);
            int delay = Integer.parseInt(f[4]), countdown = Integer.parseInt(f[5]);
            int level = Integer.parseInt(f[6]);
            Standard1v1Battle arena = new Standard1v1Battle(tables, level);
            for (int t = 0; t < 20; t++) arena.getBattle().step();
            TowerEntity tower = princess(arena, x < 9000);
            if (tower == null) throw new IllegalStateException("Missing initial Princess Tower");
            int maximum = tower.getHitPoints().getMaximum();
            tower.getHitPoints().setHitPoints(maximum * healthPercent / 100);
            int goblinHp = 0;
            for (int i = 0; i < count; i++) {
                int offset = count == 1 ? 0 : (i == 0 ? -500 : 500);
                CharacterEntity goblin = arena.deploy(20,records.unit("BushGoblin"),level,1,x+offset,y,"goblin"+i);
                goblin.getView().setDeployCountdown(countdown);
                goblinHp = goblin.getHitPoints().getHitPoints();
            }
            if (!f[7].equals("wait")) {
                arena.play(20+delay,records.card(f[7]),level,0,
                           Integer.parseInt(f[8]),Integer.parseInt(f[9]),"defender");
            }
            for (int t = 0; t < 210; t++) arena.getBattle().step();
            tower = princess(arena,x < 9000);
            System.out.println((tower == null ? 0 : Math.max(0,tower.getHitPoints().getHitPoints()))+" "+maximum+" "+goblinHp);
        }
    }
}
