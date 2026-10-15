"""Record regression references from the current code.

Run from the repo root, before a refactor:

    python -m tests.regression.make_references [--only PREFIX] [--n-jobs N]

Writes tests/regression/references.json (merging into an existing file).
With --check, compares against the references instead (in parallel, so the
slow run_sim cases finish faster than under pytest).
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use('Agg')

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), 'references.json')


def _record(name):
    from src import cellsim
    from tests.regression.cases import ALL_CASES
    cellsim.FIGURE_DIR = tempfile.mkdtemp()   # plot_view_3 always saves a file
    return name, ALL_CASES[name]()


def main():
    from tests.regression.cases import ALL_CASES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--only', default='', help='record only cases starting with this prefix')
    parser.add_argument('--n-jobs', type=int, default=os.cpu_count())
    parser.add_argument('--check', action='store_true',
                        help='compare against the references instead of recording')
    args = parser.parse_args()

    names = [n for n in ALL_CASES if n.startswith(args.only)]
    refs = {}
    if os.path.exists(REFERENCE_FILE):
        with open(REFERENCE_FILE) as f:
            refs = json.load(f)

    mismatched = []
    with ProcessPoolExecutor(args.n_jobs) as pool:
        for name, fp in pool.map(_record, names):
            if args.check:
                ok = refs.get(name) == fp
                if not ok:
                    mismatched.append(name)
                print('ok      ' if ok else 'CHANGED ', name, flush=True)
            else:
                refs[name] = fp
                print('recorded', name, flush=True)

    if args.check:
        print(f'{len(names) - len(mismatched)}/{len(names)} cases match the references')
        raise SystemExit(1 if mismatched else 0)

    with open(REFERENCE_FILE, 'w') as f:
        json.dump(dict(sorted(refs.items())), f, indent=1)
    print(f'{len(names)} cases written to {REFERENCE_FILE}')


if __name__ == '__main__':
    main()
