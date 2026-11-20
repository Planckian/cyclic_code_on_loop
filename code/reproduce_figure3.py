"""
Reproduce Figure 3 and the crossing points p_c^(M) from the CSV data in ../data.

Reads    ../data/figure3_logical_errors.csv   (logical-error counts per task)
Writes   ../data/figure3_points.csv           (basis-averaged epsilon per point, with bootstrap errors)
         ../data/figure3_crossing_points.csv  (p_c, bootstrap error, nu and the collapsed-line fit per M)
         figure3_XYZ.pdf

The analysis is the one of figure3_lut.ipynb (Section 3), which works on the raw
result files instead: epsilon per SE cycle from the fidelity fit using all rounds,
averaged over the X, Y and Z bases; p_c from the crossings of adjacent distances;
200 bootstrap resamples of every binomial count.

Usage:  python reproduce_figure3.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import lut_pipeline as lp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

DISTANCES = [5, 7, 9]
M_VALUES = [1, 2, 3]
BASES = ['X', 'Y', 'Z']
DROP_END_ROUNDS = False          # all rounds are used in the eps fit
DROP_END_P = False               # all p values are used in the crossing-point fit
N_BOOT = 200


def table_from_csv(path, in_figure3_only=True):
    """Same structure as lut_pipeline.collect_table: table[M][d][basis] = {p: (rounds, errors, shots)}."""
    df = pd.read_csv(path, float_precision='round_trip')
    if in_figure3_only:
        df = df[df.in_figure3]
    table = {}
    for (M, d, b, p), g in df.groupby(['M', 'd', 'basis', 'p']):
        g = g.sort_values('rounds')
        table.setdefault(int(M), {}).setdefault(int(d), {}).setdefault(b, {})[float(p)] = (
            g.rounds.to_numpy(), g.logical_errors.to_numpy(), g.shots.to_numpy())
    return table


# --------------------------------------------------------------------------- plotting (as in figure3_lut.ipynb)
DISTANCE_COLORS = {5: '#1B4B82', 7: '#4C9A5B', 9: '#C0392B'}
M_LINESTYLES = {1: '-', 2: '--', 3: ':'}
M_ALPHAS = {1: 1.0, 2: 0.7, 3: 0.6}
LABEL_POS = {1: (0.97, 0.70, 'right', 'center'),
             2: (0.42, 0.97, 'left', 'top'),
             3: (0.20, 0.97, 'left', 'top')}
X_TICKS = [1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
XLIM, YLIM = (1e-3, 1.1e-2), (1e-7, 1e-2)


def _xtick_label(x, _pos=None):
    e = int(np.floor(np.log10(x) + 1e-9)); m = x / 10**e
    return f'$10^{{{e}}}$' if abs(m - 1) < 1e-9 else f'${m:g}\\times10^{{{e}}}$'


def figure3(res_by_M, fname):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for M, res in res_by_M.items():
        alpha, ls = M_ALPHAS[M], M_LINESTYLES[M]
        for d in DISTANCES:
            ok = (res['eps'][d] > 1e-9) & (res['n_err'][d] > 0)     # hide points with no logical errors
            ax.errorbar(res['p'][ok], res['eps'][d][ok], yerr=res['eps_err'][d][ok], fmt='o', ms=5,
                        color=DISTANCE_COLORS[d], alpha=alpha, elinewidth=1, capsize=2)
            p_fine = np.geomspace(res['th']['p_fit'].min(), res['th']['p_fit'].max(), 300)
            ax.plot(p_fine, lp.line_curve(res, d, p_fine), linestyle=ls, color=DISTANCE_COLORS[d],
                    alpha=alpha, linewidth=1.8)
        th = res['th']
        ax.plot(th['p_th'], th['pL_th'], marker='*', markersize=20, color='black', markeredgecolor='white',
                markeredgewidth=1.0, alpha=alpha, zorder=12)
        lx, ly, lha, lva = LABEL_POS[M]
        ax.annotate(f"$p^{{({M})}}_{{c}}={lp.fmt_sig(100 * th['p_th'], 2)}\\%$", xy=(th['p_th'], th['pL_th']),
                    xytext=(lx, ly), textcoords='axes fraction', ha=lha, va=lva, fontsize=10,
                    alpha=min(alpha + 0.3, 1.0),
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', alpha=0.75 * alpha + 0.25,
                              linewidth=0.6),
                    arrowprops=dict(arrowstyle='-', color='black', alpha=alpha, linewidth=0.8), zorder=13)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.minorticks_on()
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(_xtick_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(handles=[Line2D([0], [0], color=DISTANCE_COLORS[d], linewidth=2.5, label=f'$d={d}$')
                       for d in DISTANCES], loc='lower right', fontsize=12, frameon=True)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def main():
    table = table_from_csv(os.path.join(DATA, 'figure3_logical_errors.csv'))
    res = {M: lp.analyse(table[M], DISTANCES, BASES, DROP_END_ROUNDS, DROP_END_P, N_BOOT, seed=0)
           for M in M_VALUES}

    points, crossings = [], []
    for M, r in res.items():
        for d in DISTANCES:
            for i, p in enumerate(r['p']):
                points.append(dict(M=M, d=d, p=p, logical_error_rate_per_cycle=r['eps'][d][i],
                                   bootstrap_std=r['eps_err'][d][i], logical_errors_total=int(r['n_err'][d][i])))
        th = r['th']
        crossings.append(dict(M=M, p_c=th['p_th'], p_c_bootstrap_std=r['p_th_err'],
                              p_c_ci68_low=r['p_th_ci68'][0], p_c_ci68_high=r['p_th_ci68'][1],
                              adjacent_distance_crossings=' '.join(f'{c:.6g}' for c in th['crossings']),
                              nu=th['nu'], line_a=th['poly'][1], line_b=th['poly'][0],
                              logical_error_rate_at_star=th['pL_th']))
    pd.DataFrame(points).to_csv(os.path.join(DATA, 'figure3_points.csv'), index=False, float_format='%.6g')
    cdf = pd.DataFrame(crossings)
    cdf.to_csv(os.path.join(DATA, 'figure3_crossing_points.csv'), index=False, float_format='%.6g')
    figure3(res, os.path.join(HERE, 'figure3_XYZ.pdf'))
    print(cdf[['M', 'p_c', 'p_c_bootstrap_std', 'nu']].to_string(index=False))


if __name__ == '__main__':
    main()
