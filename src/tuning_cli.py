"""Command-line entry point shared by the notebooks_2026 sweep scripts."""

import argparse
import datetime
import os

from matplotlib import pyplot as plt

from src import plotter, tuning, tuning_plots


def _results(spec_names, run, n_jobs):
    results = {}
    for name in spec_names:
        spec = tuning.SPECS[name]
        if name in run:
            result = tuning.run_sweep(spec, n_jobs=n_jobs)
            path = f'{tuning.DATA_DIR}/{name}_{datetime.date.today():%Y%m%d}.pkl'
            tuning.save_sweep(result, path)
            print(f'Saved {path}. Point SPECS[{name!r}].cache at it to use it by default.')
        else:
            result = tuning.load_sweep(spec)
        results[name] = result
    return results


def _report(result, metric, pin_col, tuned_key):
    spec = result.spec
    j = tuning.nearest_index(result.col_vec, pin_col)
    selected = tuning.select_row(result, metric, pin_col)
    print(f'\n{spec.name}: {spec.row_param} minimizing {metric} at '
          f'{spec.col_param} = {result.col_vec[j]:.4g}')
    print(f'  {"protocol":8s} {"selected":>10s} {"tuned":>10s}')
    for protocol in tuning.PROTOCOLS:
        tuned = (f'{tuning.per_protocol(tuning.TUNED[tuned_key], protocol):10.4g}'
                 if tuned_key else f'{"-":>10s}')
        print(f'  {protocol:8s} {selected[protocol]:10.4g} {tuned}')


def main(description, spec_names, pins, zlims, selections, argv=None):
    """Plot (and optionally re-run) a set of sweeps.

    pins:       {spec name: column value marked on its heatmaps}
    zlims:      {metric: (zmin, zmax)}
    selections: [(spec name, metric, TUNED key)] tuned values to report
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--run', nargs='+', choices=spec_names, default=[],
                        help='re-run these sweeps instead of loading the cached results '
                             '(takes hours)')
    parser.add_argument('--n-jobs', type=int, default=os.cpu_count(),
                        help='worker processes for --run (default: all cores)')
    parser.add_argument('--show', action='store_true', help='show the figures')
    args = parser.parse_args(argv)

    plotter.initialize(plt)
    results = _results(spec_names, args.run, args.n_jobs)

    for name, result in results.items():
        tuning_plots.plot_sweep_heatmaps(result, pin_col=pins.get(name), zlims=zlims)
        print(f'Saved heatmaps for {name} to {tuning_plots.FIGURE_DIR}/{name}_*.svg')

    for name, metric, tuned_key in selections:
        _report(results[name], metric, pins[name], tuned_key)

    if args.show:
        plt.show()
    plt.close('all')
    return results
