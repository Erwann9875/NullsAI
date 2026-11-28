from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from rl.banana_policy import BananaEpisode

from support import model_points
from policy import CARDS


CARD_COST = {
    'hog-rider': 4, 'musketeer': 4, 'ice-golem': 2, 'skeletons': 1,
    'ice-spirit': 1, 'cannon': 3, 'fireball': 4, 'the-log': 2,
}
DECK = tuple(CARD_COST)
UNKNOWN = tuple(f'unobserved-{index}' for index in range(8))
SPELLS = {'fireball', 'the-log'}
TROOP_FOOTPRINTS = ((3500, 6000), (14500, 6000), (9000, 2500))


@dataclass(frozen=True)
class Move:
    wait: bool = True
    slot: int | None = None
    card: str = ''
    x: int = 0
    y: int = 0
    reason: str = 'policy_wait'
    focus: tuple[int, int] | None = None


@lru_cache(maxsize=1)
def tile_tables():
    points = model_points({})
    x, y = points.T
    spell = (x >= 1000) & (x < 17000) & (y >= 1000)
    troop = spell & (y < 15000)
    for tower_x, tower_y in TROOP_FOOTPRINTS:
        troop &= (abs(x - tower_x) > 1450) | (abs(y - tower_y) > 1500)
    masks = {False: torch.from_numpy(troop), True: torch.from_numpy(spell)}
    return points, masks


class GreedyPolicy:
    """Single-observation decoder for the pinned v4.4.2 actor and v3 residual."""

    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        if (getattr(self.model, 'trunk', 'gru') != 'gru' or not self.model.card_conditioned_placement
                or self.model.placement_card_mode != 'selected' or self.model.placement_mode != 'heatmap'
                or getattr(self.model, 'arena_memory_channels', 0)):
            raise ValueError('Unsupported fast-path model')
        self.deck_features = None
        self.history = None
        self.history_hidden = None

    @torch.inference_mode()
    def choose(self, hand, elixir, elapsed, own_events, features, *, allowed_cards=None, tile_scores=None, seen=(), enemy_elixir=5.):
        if all(elixir < CARD_COST[card] for card in hand):
            return Move(reason='no_affordable_card')
        episode = BananaEpisode((DECK, tuple(seen)+UNKNOWN[len(seen):]), events=list(own_events),
                                battle_id='nulls-live')
        rows = [dict(slot=slot, name=card, cost=CARD_COST[card], available=elixir >= CARD_COST[card])
                for slot, card in enumerate(hand)]
        inputs, legal, mask = self.agent._inputs(episode, 0, elapsed, rows, elixir, enemy_elixir)
        if allowed_cards is not None:
            legal = [entry for entry in legal if entry[2] in allowed_cards]
            if not legal:
                return Move(reason='hold_defensive_reserve')
            mask = torch.zeros_like(mask)
            mask[[entry[1] for entry in legal]] = True
        if not legal:
            return Move(reason='no_affordable_card')
        continuous, ids, deck, opponent, globals_, slot_features = inputs[:6]
        length = int(inputs[12])
        model = self.model

        # Replay encoding is independent of affordability and the clock globals.
        # Compare the actual tensors because event time features may change.
        history = (continuous, ids, length)
        if (self.history is None or length != self.history[2]
                or not torch.equal(continuous, self.history[0]) or not torch.equal(ids, self.history[1])):
            tokens = model.input_proj(torch.cat((model.card_embedding(ids), continuous), dim=-1))
            sequence, _ = model.encoder(tokens[:max(1, length)].unsqueeze(0))
            self.history, self.history_hidden = history, sequence[:, -1]
        if self.deck_features is None or self.deck_features[0]!=tuple(seen):
            own_ids, opponent_ids = deck.unsqueeze(0), opponent.unsqueeze(0)
            mine = model.card_embedding(own_ids)
            theirs = model.card_embedding(opponent_ids)
            self.deck_features = (tuple(seen),mine, model.deck_encoder(mine, theirs, own_ids, opponent_ids))
        _,mine,deck_encoding = self.deck_features
        globals_encoding = model.global_encoder(globals_.unsqueeze(0))
        fused = model.fuse(torch.cat((self.history_hidden, deck_encoding, globals_encoding), dim=-1))
        if model.think is not None:
            fused = model.think(fused, self.agent.think_steps)
        fused = fused + self.agent.public_adapter(torch.from_numpy(features)[None])

        occupancy = features[10:].reshape(2, 3, 2, 4)
        threat = float(np.tanh(occupancy[1, 0, :, 0].sum() + .25 * occupancy[1, 1, :, 0].sum()))
        blocked_hog = float('hog-rider' in hand and elixir < CARD_COST['hog-rider'])
        blocked_musketeer = float('musketeer' in hand and elixir < CARD_COST['musketeer'])
        waiting = fused.new_tensor([1., elixir / 10, threat, blocked_hog, blocked_musketeer, blocked_hog * threat])
        adjustments = self.agent.card_adjustments
        if (allowed_cards is None or elixir<9.9) and float(self.agent.wait_head.base(fused).squeeze() + waiting @ adjustments[16:]) >= 0:
            return Move()

        slot_input = torch.cat((fused[:, None].expand(-1, 8, -1), mine, slot_features.unsqueeze(0)), dim=-1)
        scores = model.slot_scorer(slot_input).flatten()
        for _, deck_slot, card in legal:
            index = CARDS.index(card)
            scores[deck_slot] += adjustments[index] + threat * adjustments[index + 8]
        deck_slot = int(scores.masked_fill(~mask, -torch.inf).argmax())
        slot, _, card = next(row for row in legal if row[1] == deck_slot)
        placement = torch.cat((fused, mine[:, deck_slot]), dim=-1)
        zones = model.zone_head(placement).softmax(dim=-1)
        logits = model.tile_head(torch.cat((placement, zones), dim=-1)).flatten()
        points, masks = tile_tables()
        if tile_scores is not None:
            logits = logits + tile_scores(card)
        tile = int(logits.masked_fill(~masks[card in SPELLS], -torch.inf).argmax())
        x, y = (int(value) for value in points[tile])
        return Move(False, slot, card, x, y, 'policy_play')
