from types import SimpleNamespace

import pytest

import run
from native_runner.live_probe.memory_types import ExternalMemoryError, InconsistentBattleFrame


def test_busy_board_resumes_with_the_same_cursor_after_four_seconds(monkeypatch, tmp_path):
    now = [0.]
    cursors = []
    events = []
    def observe(cursor):
        cursors.append(cursor)
        now[0] += 1.55
        if len(cursors) <= 4:
            raise InconsistentBattleFrame('busy')
        return {'tick': 3189}
    monkeypatch.setattr(run.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(run.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    frame = run.observe_stable_frame(SimpleNamespace(observe=observe), 27, 20, tmp_path/'STOP',
                                     lambda name, **args: events.append(name))
    assert frame == {'tick': 3189} and cursors == [27]*5
    assert events == ['waiting_for_coherent_frame','coherent_frame_restored']


def test_inconsistent_board_stops_cleanly_and_identity_errors_propagate(tmp_path):
    stop = tmp_path/'STOP'
    stop.touch()
    def observe(cursor):
        raise InconsistentBattleFrame('busy')
    assert run.observe_stable_frame(SimpleNamespace(observe=observe), 0, float('inf'), stop, lambda *a,**k: None) is None
    def changed(cursor):
        raise ExternalMemoryError('local account identity changed')
    with pytest.raises(ExternalMemoryError, match='identity'):
        run.observe_stable_frame(SimpleNamespace(observe=changed), 0, float('inf'), stop, lambda *a,**k: None)
