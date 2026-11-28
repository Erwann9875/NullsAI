from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from support import model_points
from policy import CARDS
from rl.banana_policy import BananaEpisode


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


@dataclass(frozen=True)
class Threat:
    x: int
    y: int


def visible_threats(image):
    crop = np.asarray(image.crop((85, 405, 815, 1215)))[::2, ::2, :3]
    red, green, blue = crop.transpose(2, 0, 1)
    mask = (red >= 76) & (red <= 119) & (green <= 13) & (blue >= 18) & (blue <= 46)
    padded = np.pad(mask, ((0, 0), (1, 1)))
    edges = padded[:, 1:] ^ padded[:, :-1]
    rows, columns = edges.nonzero()
    widths = columns[1::2] - columns[::2]
    rows = rows[::2]
    starts = columns[::2]
    usable = (widths >= 8) & (widths <= 52)
    candidates = []
    for row, start, width in zip(rows[usable], starts[usable], widths[usable]):
        center = float(start + width / 2)
        group = next((g for g in reversed(candidates)
                      if row - g['last'] <= 7 and abs(center - g['center']) <= 8), None)
        if group is None:
            candidates.append(dict(first=row, last=row, center=center, samples=1))
        else:
            group['last'] = row
            group['samples'] += 1
    known = []
    for group in candidates:
        height = group['last'] - group['first'] + 1
        if group['samples'] < 2 or not 2 <= height <= 13:
            continue
        pixel_x = 85 + 2 * group['center']
        pixel_y = 405 + 2 * (group['first'] + group['last']) / 2 + 30
        x, y = screen_to_board(pixel_x, pixel_y)
        if any(abs(x - other.x) <= 950 and abs(y - other.y) <= 1200 for other in known):
            continue
        known.append(Threat(x, y))
    return tuple(known)


def screen_to_board(x, y):
    return (round(np.clip((x - 75) * 18000 / 750, 0, 17999)),
            round(np.clip((1200 - y) * 32000 / 1040, 0, 31999)))


def features_from_threats(threats):
    features = np.zeros(58, dtype=np.float32)
    bins = features[10:].reshape(2, 3, 2, 4)
    for threat in threats:
        bins[1, min(2, threat.y * 3 // 32000), int(threat.x >= 9000), 0] += .2
    np.clip(bins, 0, 2, out=bins)
    return features


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


def defense_move(hand, elixir, threats, past_defense, now):
    incoming = tuple(t for t in threats if 800 <= t.y <= 16300)
    if not incoming:
        return None
    priority = min(incoming, key=lambda t: (t.y, abs(t.x - 9000)))
    lane = int(priority.x >= 9000)
    if past_defense is not None:
        previous, previous_lane, previous_y = past_defense
        if lane == previous_lane and now - previous < 4 and priority.y >= previous_y - 2500:
            return Move(reason='defender_recently_deployed')

    ready = {card: slot for slot, card in enumerate(hand) if elixir >= CARD_COST[card]}
    cluster = tuple(t for t in incoming if math.hypot(t.x - priority.x, t.y - priority.y) <= 3200)
    if len(cluster) >= 2 and 'fireball' in ready:
        card = 'fireball'
        aim_x = round(sum(t.x for t in cluster) / len(cluster))
        aim_y = round(sum(t.y for t in cluster) / len(cluster))
    elif 'musketeer' in ready:
        card = 'musketeer'
        aim_x = priority.x + (1200 if lane == 0 else -1200)
        aim_y = priority.y - 3700
    elif 'cannon' in ready and priority.y >= 7000:
        card = 'cannon'
        aim_x = 7100 if lane == 0 else 10900
        aim_y = min(11200, max(7200, priority.y - 3000))
    elif priority.y < 10500 and 'ice-spirit' in ready:
        card = 'ice-spirit'
        aim_x, aim_y = priority.x, priority.y - 1700
    elif priority.y < 6500 and 'skeletons' in ready:
        card = 'skeletons'
        aim_x, aim_y = priority.x, priority.y - 1100
    else:
        return Move(reason='save_elixir_for_defense')

    points, masks = tile_tables()
    legal = masks[card in SPELLS].numpy()
    distance = (points[:, 0] - aim_x) ** 2 + (points[:, 1] - aim_y) ** 2
    tile = int(np.where(legal, distance, np.inf).argmin())
    x, y = (int(value) for value in points[tile])
    return Move(False, ready[card], card, x, y, 'defend_visible_troops', (lane, priority.y))


class GreedyPolicy:

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
    def choose(self, hand, elixir, elapsed, own_events, features):
        if all(elixir < CARD_COST[card] for card in hand):
            return Move(reason='no_affordable_card')
        episode = BananaEpisode((DECK, UNKNOWN), events=list(own_events),
                                battle_id='nulls-practice-screen')
        rows = [dict(slot=slot, name=card, cost=CARD_COST[card], available=elixir >= CARD_COST[card])
                for slot, card in enumerate(hand)]
        inputs, legal, mask = self.agent._inputs(episode, 0, elapsed, rows, elixir, 5.)
        if not legal:
            return Move(reason='no_affordable_card')
        continuous, ids, deck, opponent, globals_, slot_features = inputs[:6]
        length = int(inputs[12])
        model = self.model

        history = (continuous, ids, length)
        if (self.history is None or length != self.history[2]
                or not torch.equal(continuous, self.history[0]) or not torch.equal(ids, self.history[1])):
            tokens = model.input_proj(torch.cat((model.card_embedding(ids), continuous), dim=-1))
            sequence, _ = model.encoder(tokens[:max(1, length)].unsqueeze(0))
            self.history, self.history_hidden = history, sequence[:, -1]
        if self.deck_features is None:
            own_ids, opponent_ids = deck.unsqueeze(0), opponent.unsqueeze(0)
            mine = model.card_embedding(own_ids)
            theirs = model.card_embedding(opponent_ids)
            self.deck_features = (mine, model.deck_encoder(mine, theirs, own_ids, opponent_ids))
        mine, deck_encoding = self.deck_features
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
        if float(self.agent.wait_head.base(fused).squeeze() + waiting @ adjustments[16:]) >= 0:
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
        tile = int(logits.masked_fill(~masks[card in SPELLS], -torch.inf).argmax())
        x, y = (int(value) for value in points[tile])
        return Move(False, slot, card, x, y, 'policy_play')
