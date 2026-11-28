from types import SimpleNamespace

from memory_policy import MemoryPolicy


def test_hand_rotation_can_precede_the_elixir_debit():
    policy = object.__new__(MemoryPolicy)
    policy.elixir_cap = 1.2
    policy.elixir_tick = 354
    policy.balance_settles_at = 0
    policy.recent = None
    policy.submitted_focus = {354: None}
    action = SimpleNamespace(card_id=28_000_000)
    pending = SimpleNamespace(requested_tick=354, action=action)
    policy.builder = SimpleNamespace(pending=[], receipts=[(pending, 'confirmed')])
    policy.inputs = SimpleNamespace(card=lambda card: 'fireball')
    own = SimpleNamespace(elixir_exact=5.7284)
    obs = SimpleNamespace(tick=380, players=[own], time=SimpleNamespace(elixir_multiplier=1.))
    policy.record_receipts(obs, lambda *args, **kwargs: None)
    policy.update_elixir(obs)
    assert 1.6 < policy.elixir_cap < 1.7
    own.elixir_exact, obs.tick = 1.9, 387
    policy.update_elixir(obs)
    assert policy.elixir_cap <= own.elixir_exact
    own.elixir_exact, obs.tick = 2.1, 402
    policy.update_elixir(obs)
    assert policy.elixir_cap == own.elixir_exact
