"""Tables from fixed-feature cell JSONs: cosines to the unrolled gradient and spectra,
outer-loop accuracy, and the closed-form baseline with its temperature selected on
validation.

    python scripts/ladder_tables.py runs/headladder
"""
import argparse
import collections
import glob
import json
import os

EST_CERT = ('identity', 'cg10', 'cg100', 'neumann10', 'neumann100', 'minres', 'damped',
            'nystrom10', 'nystrom50', 'dense')


def stat(xs, fmt='{:.3f}'):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '–'
    m = sum(xs) / len(xs)
    s = fmt.format(m)
    if len(xs) > 1 and max(xs) - min(xs) > 0:
        s += f" ({fmt.format(min(xs))}..{fmt.format(max(xs))})"
    return s


def key(d):
    a = d['args']
    return (a['head'], a['ridge'], a.get('val_per_class', 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    args = ap.parse_args()
    cells = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.dir, '*.json')))]
    cells = [d for d in cells if d.get('status') == 'ok']
    cert = collections.defaultdict(list)
    loop = collections.defaultdict(list)
    ref = collections.defaultdict(list)
    for d in cells:
        a = d['args']
        k = key(d)
        if a['mode'] == 'cert':
            cert[k + (a['point'], a['k_list'], a['polish_iters'], a['damp_rel'])].append(d)
        elif a['mode'] == 'loop':
            loop[k + (a['point'], a['estimator'])].append(d)
        else:
            ref[k].append(d)

    print("## cert: cosine of each estimator's hypergradient to the unrolled (finite-difference) gradient (mean, min..max over seeds)")
    ests = [e for e in EST_CERT if any(e in d.get('estimators', {}) for ds in cert.values() for d in ds)]
    print("| head | ridge | val | point | k | polish | damp | n | lam_min | n_neg | " + ' | '.join(ests) + " |")
    print("|" + "---|" * (10 + len(ests)))
    for k in sorted(cert):
        ds = cert[k]
        sp = [d.get('spectrum', {}) for d in ds]
        row = [k[0], f"{k[1]:g}", str(k[2]), k[3], k[4], str(k[5]), f"{k[6]:g}", str(len(ds)),
               stat([s.get('lam_min') for s in sp], '{:.1e}'), stat([s.get('n_neg') for s in sp], '{:.0f}')]
        row += [stat([d['estimators'][e]['cos_fd'] for d in ds if e in d['estimators']], '{:.2f}') for e in ests]
        print("| " + " | ".join(row) + " |")

    print("\n## ref: logit adjustment l = tau log(pi) trained on this head (balanced test accuracy)")
    print("| head | ridge | val | n | CE (tau=0) | tau=1 | tau=2 | best tau by val -> test | best tau by test (oracle) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(ref):
        ds = ref[k]
        by_val, by_test = [], []
        for d in ds:
            r = d['ref']
            if all(v.get('bacc_val') is not None for v in r.values()):
                tv = max(r, key=lambda t: r[t]['bacc_val'])
                by_val.append(r[tv]['bacc'])
            by_test.append(max(v['bacc'] for v in r.values() if v['bacc'] is not None))
        g = lambda t: [d['ref'][t]['bacc'] for d in ds if t in d['ref']]
        print(f"| {k[0]} | {k[1]:g} | {k[2]} | {len(ds)} | {stat(g('0.0'))} | {stat(g('1.0'))} | {stat(g('2.0'))} | "
              f"{stat(by_val)} | {stat(by_test)} |")

    print("\n## loop: final balanced test accuracy per estimator (mean, min..max over seeds)")
    ests_l = sorted({k[4] for k in loop})
    starts = sorted({k[3] for k in loop})
    print("| head | ridge | val | start | " + ' | '.join(ests_l) + " |")
    print("|" + "---|" * (4 + len(ests_l)))
    for hk in sorted({k[:3] for k in loop}):
        for st in starts:
            row = [hk[0], f"{hk[1]:g}", str(hk[2]), st]
            row += [stat([d['final_bacc'] for d in loop.get(hk + (st, e), [])]) for e in ests_l]
            print("| " + " | ".join(row) + " |")


if __name__ == '__main__':
    main()
