"""Table of end-to-end runs by arm and validation size, with the closed-form baseline's
temperature selected on the same validation set the bilevel leader trained on.

    python scripts/autobalance_tables.py runs/autobalance

Arms: ce; la tau=1 (no validation used); la-val (tau chosen per seed by validation balanced
accuracy among the trained tau grid); la-oracle (tau chosen on test, an upper bound, not a
method); every ab-* configuration found. Cells: mean +- std (n) of balanced test accuracy.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def fmt(xs):
    if not xs:
        return '–'
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f} (n={len(xs)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--metric', default='bacc', choices=('bacc', 'acc'))
    args = ap.parse_args()
    runs = []
    for f in glob.glob(os.path.join(args.dir, '*', 'metrics.json')):
        d = json.load(open(f))
        runs.append(d)
    if not runs:
        print("no runs")
        return
    m = args.metric
    vals = sorted({d['args']['val_per_class'] for d in runs})
    imbs = sorted({d['args']['imbalance'] for d in runs})
    for imb in imbs:
        print(f"## imbalance {imb}: balanced test accuracy ({m}) by validation images per class")
        print("| arm | " + " | ".join(f"val {v}" for v in vals) + " |")
        print("|---|" + "---|" * len(vals))
        rows = collections.OrderedDict()
        for v in vals:
            sel = [d for d in runs if d['args']['imbalance'] == imb and d['args']['val_per_class'] == v]
            la = collections.defaultdict(dict)          # seed -> tau -> run
            for d in sel:
                a = d['args']
                if a['method'] == 'ce':
                    rows.setdefault('ce', {})[v] = rows.get('ce', {}).get(v, []) + [d['final']['test'][m]]
                elif a['method'] == 'la':
                    la[a['seed']][a['tau']] = d
                    rows.setdefault(f"la tau={a['tau']:g}", {}).setdefault(v, []).append(d['final']['test'][m])
                else:
                    name = d['name'].rsplit('-imb', 1)[0]
                    rows.setdefault(name, {}).setdefault(v, []).append(d['final']['test'][m])
            for seed, taus in la.items():
                if len(taus) < 2:
                    continue
                by_val = max(taus.values(), key=lambda d: d['final']['val']['bacc'])
                by_test = max(taus.values(), key=lambda d: d['final']['test'][m])
                rows.setdefault('la-val (tau by val bacc)', {}).setdefault(v, []).append(by_val['final']['test'][m])
                rows.setdefault('la-oracle (tau by test)', {}).setdefault(v, []).append(by_test['final']['test'][m])
        for name in sorted(rows, key=lambda n: (not n.startswith('ce'), not n.startswith('la'), n)):
            print(f"| {name} | " + " | ".join(fmt(rows[name].get(v, [])) for v in vals) + " |")
        print()


if __name__ == '__main__':
    main()
