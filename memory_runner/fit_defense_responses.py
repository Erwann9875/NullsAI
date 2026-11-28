"""Learn Hog-cycle responses to earlier opponent deployments from match-split IL_Replay."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from fit_tile_prior import CARDS


def family(name):
    return re.sub(r'-(?:ev\d+|hero)$', '', name)


def tally(cache, catalog):
    manifest = json.loads((cache / 'manifest.json').read_text())
    vocab = manifest['cards']
    threat_names = sorted({record['id'].replace('_', '-') for record in catalog
                           if record['kind'] == 'troop'} | {'x-bow', 'mortar', 'goblin-barrel'})
    own = np.full(max(vocab.values()) + 1, -1, dtype=np.int16)
    enemy = np.full_like(own, -1)
    for name, index in vocab.items():
        name = family(name)
        if name in CARDS:
            own[index] = CARDS.index(name)
        if name in threat_names:
            enemy[index] = threat_names.index(name)
    hist = np.zeros((3, len(threat_names), 3, len(CARDS), 32, 18), dtype=np.int32)
    matches = Counter()
    for path in sorted(cache.glob('*.npz')):
        with np.load(path, allow_pickle=False) as shard:
            decks = own[shard['decks']]
            hog = (decks == 0).any(2) & (decks == 5).any(2) & ((decks >= 0).sum(2) >= 6)
            if not hog.any():
                continue
            data, counts, splits = shard['events'], shard['counts'], shard['splits']
            for side in (0, 1):
                subset = hog[:, side]
                if not subset.any():
                    continue
                ids, ticks, owners, x, y, slots, prefix = data[subset].astype(np.int32).transpose(2, 0, 1)
                if side:
                    y = 31999 - y
                sequence = np.arange(ids.shape[1])[None, :]
                valid = (sequence < counts[subset, None]) & (slots >= 0) & (x >= 0)
                opposing = valid & (owners != side) & (enemy[ids] >= 0)
                last = np.maximum.accumulate(np.where(opposing, sequence, -1), axis=1)
                idx = np.take_along_axis(last, np.maximum(0, prefix-1), axis=1)
                earlier = np.maximum(idx, 0)
                def prev(values):
                    return np.take_along_axis(values, earlier, axis=1)
                age = ticks - prev(ticks)
                pick = valid & (owners == side) & (own[ids] >= 0) & (prefix > 0) & (idx >= 0) & (age > 0) & (age <= 160)
                depth = (prev(y) > 13000).astype(int) + (prev(y) > 21000)
                col = np.where(prev(x) < 9000, x // 1000, 17-x//1000)
                split = np.broadcast_to(splits[subset, None], ids.shape)
                arrays = (split, enemy[prev(ids)], depth, own[ids], y // 1000, col)
                np.add.at(hist, tuple(a[pick] for a in arrays), 1)
                matches.update(map(int, splits[subset][pick.any(1)]))
    return threat_names, hist, matches, manifest['dataset_revision']


def norm(values, axes):
    return values / values.sum(axis=axes, keepdims=True)


def spatial_smooth(values):
    for axis in (-1, -2):
        pad = [(0, 0)] * values.ndim
        pad[axis] = (1, 1)
        padded = np.pad(values, pad, mode='edge')
        low = np.take(padded, np.arange(values.shape[axis]), axis=axis)
        high = np.take(padded, 2 + np.arange(values.shape[axis]), axis=axis)
        values = .2 * low + .6 * values + .2 * high
    return values


def fit(train, strength):
    counts = train.sum(axis=(-2, -1))
    base_card = norm(counts.sum(axis=0, keepdims=True) + 1., -1)
    enemy_card = norm(counts.sum(axis=1, keepdims=True) + strength * base_card.mean(1, keepdims=True), -1)
    card = norm(counts + strength * (enemy_card + base_card) / 2, -1)
    base_tile = norm(spatial_smooth(train.sum(axis=0, keepdims=True)) + .04, (-2, -1))
    enemy_tile = norm(spatial_smooth(train.sum(axis=1, keepdims=True)) +
                      strength * base_tile.mean(1, keepdims=True), (-2, -1))
    tile = norm(spatial_smooth(train) + strength * (enemy_tile + base_tile) / 2, (-2, -1))
    return card, tile, base_card, base_tile


def metrics(hist, card, tile):
    counts = hist.sum(axis=(-2, -1))
    card = np.broadcast_to(card, counts.shape)
    tile = np.broadcast_to(tile, hist.shape)
    n = int(hist.sum())
    chosen = np.take_along_axis(counts, card.argmax(-1)[..., None], axis=-1)
    rows, cols = np.divmod(tile.reshape(*counts.shape, -1).argmax(-1), 18)
    y, x = np.indices((32, 18))
    distance = np.hypot(y-rows[..., None, None], x-cols[..., None, None])
    return dict(examples=n, card_accuracy=float(chosen.sum()/n),
                card_nll=float(-(counts*np.log(card)).sum()/n),
                tile_nll=float(-(hist*np.log(tile)).sum()/n),
                tile_mode_error=float((hist*distance).sum()/n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    from support import limit_cpu
    limit_cpu()
    names, hist, matches, revision = tally(args.cache, json.loads(args.catalog.read_text(encoding='utf8')))
    train, valid, test = hist
    validation = {}
    for strength in (12, 48, 160):
        card, tile, _, _ = fit(train, strength)
        validation[strength] = metrics(valid, card, tile)
    best = min(validation, key=lambda k: validation[k]['card_nll'] + validation[k]['tile_nll'])
    card, tile, base_card, base_tile = fit(train, best)
    model_metrics = metrics(test, card, tile)
    baseline_metrics = metrics(test, base_card, base_tile)
    report = dict(dataset_revision=revision, strengths=validation, selected=best,
                  matching_deck_sides={label:int(matches[i]) for i,label in enumerate(('train','validation','test'))},
                  examples=[int(h.sum()) for h in hist], heldout=model_metrics, baseline=baseline_metrics,
                  baseline_description='Same depth, lane, response card and smoothing; opponent identity withheld',
                  per_opponent={name:metrics(test[i:i+1],card[i:i+1],tile[i:i+1]) for i,name in enumerate(names)
                                if test[i].sum() >= 10})
    args.report.write_text(json.dumps(report,indent=2)+'\n', encoding='utf8')
    np.savez_compressed(args.output, cards=np.asarray(CARDS), enemies=np.asarray(names),
                        card_logp=np.log(card).astype('float32'), tile_logp=np.log(tile).astype('float32'),
                        support=train.sum(axis=(2,3,4)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_opponent','strengths')}, indent=2))


if __name__ == '__main__':
    main()
