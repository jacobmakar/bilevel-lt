"""Live runtime estimate for a ladder sweep.

    python3 scripts/sweep_eta.py <results_dir> [--par N]

<results_dir> holds jobs/*.txt (the cell command lines) and sweep-*.log (written by
slurm/jobs.sbatch: START/END stamps per cell, PROGRESS lines from bilevel_lt/ladder.py,
and the one-line JSON summary each cell prints at the end).

Estimates: cells done / running / pending; mean duration per (mode, head, estimator)
from finished cells; remaining work = predicted durations of pending cells + remaining
time of running cells (their own PROGRESS eta when present); wall ETA = remaining work /
observed parallelism, plus a throughput-based cross-check. Stdlib only, so it runs with
the system python3 on the cluster.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilevel_lt.tags import cell_tag  # noqa: E402  (torch-free)


def parse_cmd(cmd):
    """`--name value` pairs of a cell command line; defaults come from bilevel_lt.tags."""
    toks = cmd.split()
    a = {}
    for i, t in enumerate(toks):
        if t.startswith('--') and i + 1 < len(toks) and not toks[i + 1].startswith('--'):
            a[t[2:]] = toks[i + 1]
    return a


def tag_of(a):
    return cell_tag(a)


def norm(cmd):
    cmd = re.sub(r'--out_root \S+', '', cmd)
    cmd = re.sub(r'--threads \S+', '', cmd)
    return ' '.join(cmd.split())


def fmt_dur(s):
    if s is None:
        return '–'
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir')
    ap.add_argument('--par', type=int, default=None,
                    help="assumed parallelism if no cell is currently running")
    args = ap.parse_args()
    root = os.path.expanduser(args.results_dir)
    now = time.time()

    cells = {}                       # norm cmd -> parsed args
    for f in sorted(glob.glob(os.path.join(root, 'jobs', '*.txt'))):
        for line in open(f):
            line = line.strip()
            if line:
                cells[norm(line)] = parse_cmd(line)
    by_tag = {tag_of(a): c for c, a in cells.items()}

    start, end, rc, progress, json_sec = {}, {}, {}, {}, {}
    start_log = {}
    live_logs = set()                # logs modified recently: their jobs are still alive
    first_start = None
    for f in sorted(glob.glob(os.path.join(root, 'sweep-*.log'))):
        if now - os.path.getmtime(f) < 5400:     # dense loop arm prints PROGRESS only every ~30 min
            live_logs.add(f)
        for line in open(f, errors='replace'):
            line = line.rstrip('\n')
            if line.startswith('START '):
                _, ts, cmd = line.split(' ', 2)
                c = norm(cmd); start[c] = float(ts); start_log[c] = f
                first_start = min(first_start or float(ts), float(ts))
            elif line.startswith('END '):
                m = re.match(r'END (\d+) rc=(\d+) (.*)', line)
                if m:
                    c = norm(m.group(3)); end[c] = float(m.group(1)); rc[c] = int(m.group(2))
            elif line.startswith('PROGRESS '):
                parts = line.split()
                tag = parts[1]
                kv = dict(p.split('=', 1) for p in parts[2:] if '=' in p)
                progress[tag] = kv
            else:
                m = re.match(r'(\S+) (\{"status".*\})$', line)
                if m:
                    try:
                        json_sec[m.group(1)] = json.loads(m.group(2)).get('seconds')
                    except json.JSONDecodeError:
                        pass

    done, running, pending, failed = {}, {}, [], []
    for c, a in cells.items():
        tag = tag_of(a)
        if c in end:
            dur = end[c] - start.get(c, end[c])
            if rc.get(c, 0) != 0:
                failed.append(c)
            done[c] = dur
        elif tag in json_sec and json_sec[tag] is not None:   # logs without START/END stamps
            done[c] = json_sec[tag]
        elif c in start and start_log[c] in live_logs:
            running[c] = now - start[c]
        else:                       # never started, or started in a job that died (rerun)
            pending.append(c)

    # duration model: (mode, head, estimator) -> (mode, head) -> mode -> global
    groups = [defaultdict(list) for _ in range(4)]
    keyf = [lambda a: (a.get('mode', 'cert'), a['head'], a.get('estimator'), a.get('k_list'), a.get('polish_iters'), a.get('outer_steps')),
            lambda a: (a.get('mode', 'cert'), a['head']), lambda a: (a.get('mode', 'cert'),), lambda a: ()]
    for c, dur in done.items():
        for g, kf in zip(groups, keyf):
            g[kf(cells[c])].append(dur)

    def predict(a):
        for g, kf in zip(groups, keyf):
            xs = g.get(kf(a))
            if xs:
                return sum(xs) / len(xs), len(xs)
        return None, 0

    remaining_pending = 0.0
    unknown = 0
    for c in pending:
        p, n = predict(cells[c])
        if p is None:
            unknown += 1
        else:
            remaining_pending += p
    remaining_running = 0.0
    run_rows = []
    for c, el in sorted(running.items(), key=lambda kv: -kv[1]):
        a = cells[c]; tag = tag_of(a); pr = progress.get(tag, {})
        eta = None
        # a cell's own ETA is trusted only once it is a few percent in: the first outer
        # step's estimate includes the warm start and overshoots by 2-3x
        step_frac = 1.0
        if 'step' in pr and '/' in pr['step']:
            try:
                a_, b_ = pr['step'].split('/')
                step_frac = int(a_) / max(int(b_), 1)
            except ValueError:
                pass
        if step_frac >= 0.05:
            for k in ('eta', 'eta_fd'):
                if k in pr:
                    try:
                        eta = float(pr[k])
                        # the PROGRESS line may be old (dense arm prints every 20 steps):
                        # subtract the time elapsed since it was printed
                        eta = max(0.0, eta - (el - float(pr.get('elapsed', el))))
                    except ValueError:
                        pass
        if eta is None:
            p, _ = predict(a)
            eta = max(0.0, (p or 0.0) - el)
        remaining_running += eta
        run_rows.append((tag, el, eta, ' '.join(f"{k}={v}" for k, v in pr.items() if k not in ('elapsed',))))

    par = len(running) or args.par or 8
    total = len(cells)
    print(f"# {root}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"cells: {total} total | {len(done)} done ({len(failed)} failed) | {len(running)} running | {len(pending)} pending"
          + (f" ({unknown} with no duration model yet)" if unknown else ''))
    if done:
        print("\nmean duration of finished cells by mode/head (n):")
        rows = sorted(groups[1].items())
        for (mode, head), xs in rows:
            print(f"  {mode:5s} {head:10s} {fmt_dur(sum(xs) / len(xs)):>8s}  (n={len(xs)}, max {fmt_dur(max(xs))})")
    if running:
        print("\nrunning now:")
        for tag, el, eta, pr in run_rows:
            print(f"  {tag:45s} elapsed {fmt_dur(el):>7s}  eta {fmt_dur(eta):>7s}  {pr}")
    work = remaining_pending + remaining_running
    longest = max((eta for _, _, eta, _ in run_rows), default=0.0)
    wall = max(work / par, longest)          # critical path: the slowest running cell
    print(f"\nremaining work: {fmt_dur(work)} cell-time over parallelism {par} = {fmt_dur(work / par)}; "
          f"longest running cell {fmt_dur(longest)}"
          f"  ->  wall ETA ~{fmt_dur(wall)}  (finish ~{time.strftime('%H:%M', time.localtime(now + wall))})")
    if first_start and len(done) >= 3:
        hours = (now - first_start) / 3600
        thr = len(done) / hours
        print(f"throughput cross-check: {thr:.1f} cells/h since first START -> pending+running "
              f"{len(pending) + len(running)} cells ~{fmt_dur((len(pending) + len(running)) / thr * 3600)}")
    if failed:
        print("\nFAILED cells (rc != 0):")
        for c in failed:
            print("  " + tag_of(cells[c]))


if __name__ == '__main__':
    main()
