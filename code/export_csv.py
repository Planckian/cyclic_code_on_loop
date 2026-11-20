"""
Export the Figure 3 simulation results from the raw result files (lut_results/*.pkl,
written by lut_pipeline.run_tasks) to the CSV file in ../data.

Only the records used in the paper are exported: decoder HybridLUT, M = 1, 2, 3,
d = 5, 7, 9, bases X, Y, Z, r = 1, and (for M > 1) idle noise on the syndrome qubits
still waiting to be measured ('waiting').  These are the same selection rules as
lut_pipeline.collect_table.  Several records of the same task (top-up runs) are
independent samples of the same circuit and are summed.

Usage:  python export_csv.py [results_dir] [output_csv]
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

import lut_pipeline as lp

M_VALUES = [1, 2, 3]
DISTANCES = [5, 7, 9]
BASES = ['X', 'Y', 'Z']
R_IDLE = 1.0
DECODER = 'HybridLUT'
ANCILLAS = 'waiting'

# p values simulated but left out of Figure 3 (same as P_EXCLUDE in figure3_lut.ipynb)
P_VALUES = np.logspace(np.log10(1e-3), np.log10(8e-3), num=10)
P_EXTRA_M1 = P_VALUES[-1] * (P_VALUES[1] / P_VALUES[0]) ** np.arange(1, 5)
P_EXCLUDE = {1: P_EXTRA_M1[1:], 2: P_VALUES[-1:], 3: P_VALUES[-1:]}


class _Unpickler(pickle.Unpickler):
    """Load records pickled with other stim builds (module stim._stim_<arch> -> stim)."""
    def find_class(self, module, name):
        if module.startswith('stim._stim_'):
            module = 'stim'
        return super().find_class(module, name)


def records(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield _Unpickler(f).load()
            except EOFError:
                return
            except pickle.UnpicklingError:      # truncated last record of an interrupted run
                return


def export(results_dir='lut_results', out='../data/figure3_logical_errors.csv'):
    acc = {}
    for M in M_VALUES:
        for d in DISTANCES:
            for basis in BASES:
                for rec in records(lp.group_file(results_dir, M, R_IDLE, d, basis)):
                    if rec.decoder != DECODER:
                        continue
                    if M > 1 and getattr(rec.code, 'readout_noise_ancillas', None) != ANCILLAS:
                        continue
                    p = float(rec.code.before_round_data_depolarization)
                    key = (M, d, int(rec.code.k), basis, p, int(rec.code.rounds))
                    a = acc.setdefault(key, [0, 0, 0])
                    a[0] += int(rec.shots); a[1] += int(rec.errors); a[2] += int(getattr(rec, 'fallback', 0))
    rows = []
    for (M, d, k, basis, p, R), (shots, errors, fallback) in sorted(acc.items()):
        rows.append(dict(M=M, d=d, measurement_sites=k, basis=basis, p=p, rounds=R,
                         shots=shots, logical_errors=errors, fallback_shots=fallback,
                         in_figure3=not np.any(np.isclose(p, P_EXCLUDE[M], rtol=1e-6))))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{len(df)} rows -> {out}')
    return df


if __name__ == '__main__':
    export(*sys.argv[1:3])
