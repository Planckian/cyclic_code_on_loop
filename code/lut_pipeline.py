"""
Running and analysing the lookup-table (BetterLUT) memory experiments for Figure 3.

Simulation
----------
make_code / build_tasks / run_tasks run tasks in a process pool and append each
finished TaskStats to <results_dir>/M<M>_r<r>_d<d>_<basis>.pkl immediately, so an
interrupted run resumes where it stopped.

Analysis (follows plot_results_phenom.ipynb, Figure 3 cell)
---------------------------------------------------------
* epsilon per cycle from F(R) = (1 + F0 (1 - 2 eps)^R) / 2 with the ORIGINAL fit
  (fit_fidelity_vs_rounds: unweighted, same bounds); drop_end_rounds=True reproduces
  the original behaviour of dropping the first and last round when there are > 3.
* epsilon is averaged over bases (mean of the fitted epsilon).
* crossing point: fss_crossing.estimate_crossing_point (median of adjacent-distance
  crossings), then the collapsed straight-line fit log(eps) = a + b x,
  x = log(p / p_c) d^(1/nu), whose lines for all distances meet at (p_c, exp(a)).
  drop_end_p=True excludes the lowest and highest p from this fit.
* bootstrap: every binomial count is resampled, epsilon refitted, and the whole
  crossing-point analysis repeated, giving error bars on epsilon and on p_c.
"""
import os
import time
import pickle
import warnings
from multiprocessing import Pool

import numpy as np

from ChainCodeGlobal import ChainCodeGlobal
from LinearChainCodeGlobal import LinearChainCodeGlobal
from collectorGlobal import single_run, prepare_luts
from process_data import task_key, fit_fidelity_vs_rounds
from fss_crossing import estimate_crossing_point, _fss_scaling_variable


# --------------------------------------------------------------------------- codes and tasks
def code_class(d):
    return ChainCodeGlobal if d in (3, 5) else LinearChainCodeGlobal


def make_code(d, M, r, p, basis, rounds, ancillas='waiting'):
    """Phenomenological model: data depolarizing p per cycle, measurement flip q = p,
    r*p on data (and on syndrome qubits per `ancillas`) after each extra readout round,
    no gate or reset noise.  k = smallest number of measurement sites giving M rounds."""
    cls = code_class(d)
    return cls(distance=d, rounds=int(rounds), basis=basis, k=cls.k_for_rounds(d, M),
               before_round_data_depolarization=float(p), before_measure_flip_probability=float(p),
               after_clifford_depolarization_Q1=0., after_clifford_depolarization_Q2=0.,
               after_reset_flip_probability=0., readout_noise_ratio=float(r),
               readout_noise_ancillas=ancillas, native_2q_gate_set=['iSWAP'])


def group_file(results_dir, M, r, d, basis):
    return os.path.join(results_dir, f'M{M}_r{r:g}_d{d}_{basis}.pkl')


def load_records(path):
    """Stream of pickled TaskStats; a truncated last record (crash) is ignored."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'rb') as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
            except Exception:
                break
    return out


def _append(path, record):
    with open(path, 'ab') as f:
        pickle.dump(record, f)
        f.flush()
        os.fsync(f.fileno())


def build_tasks(M_values, r, distances, bases, p_values, rounds, results_dir, ancillas='waiting'):
    """(file, code) pairs, ordered by (distance, basis) so that consecutive tasks of a
    worker reuse the same lookup table."""
    tasks = []
    for d in sorted(distances, reverse=True):
        for basis in bases:
            for M in M_values:
                path = group_file(results_dir, M, r, d, basis)
                for p in p_values:
                    for R in rounds:
                        tasks.append((path, make_code(d, M, r, p, basis, R, ancillas)))
    return tasks


def pending(tasks, decoder):
    done, out = {}, []
    for path, code in tasks:
        if path not in done:
            done[path] = {rec.key for rec in load_records(path)}
        if task_key(code, decoder) not in done[path]:
            out.append((path, code))
    return out


def _run_one(args):
    path, code, max_shots, max_errors, shots_step, decoder = args
    return path, single_run(code, max_shots, max_errors, [decoder], shots_step)[0]


def existing_counts(tasks, decoder):
    """Total (errors, shots) already on disk for every task (summed over top-up records)."""
    cache, out = {}, []
    for path, code in tasks:
        if path not in cache:
            tot = {}
            for rec in load_records(path):
                e, s = tot.get(rec.key, (0, 0))
                tot[rec.key] = (e + int(rec.errors), s + int(rec.shots))
            cache[path] = tot
        out.append(cache[path].get(task_key(code, decoder), (0, 0)))
    return out


def topup_tasks(tasks, max_shots, max_errors, workers, shots_step=5000, decoder='BetterLUT'):
    """Add shots to tasks until each has max_errors errors OR max_shots shots IN TOTAL
    (existing + new).  New results are appended to the same files as extra records;
    collect_table adds them up.  Tasks never run before are run from scratch."""
    counts = existing_counts(tasks, decoder)
    todo = []
    for (path, code), (e, s) in zip(tasks, counts):
        rem_e, rem_s = max_errors - e, max_shots - s
        if rem_e > 0 and rem_s > 0:
            todo.append((path, code, rem_s, rem_e))
    print(f'{len(tasks)} tasks: {len(tasks) - len(todo)} already at the new target, {len(todo)} to top up')
    if not todo:
        return
    os.makedirs(os.path.dirname(todo[0][0]) or '.', exist_ok=True)
    prepare_luts([code for _, code, _, _ in todo])
    args = [(path, code, rem_s, rem_e, shots_step, decoder) for path, code, rem_s, rem_e in todo]
    t0 = time.time()
    from tqdm.auto import tqdm
    if workers == 1:
        for path, res in tqdm((_run_one(a) for a in args), total=len(args)):
            _append(path, res)
    else:
        with Pool(workers) as pool:
            for path, res in tqdm(pool.imap(_run_one, args, chunksize=1), total=len(args)):
                _append(path, res)
    print(f'finished in {(time.time() - t0) / 3600:.2f} h')


def run_tasks(tasks, max_shots, max_errors, workers, shots_step=5000, decoder='BetterLUT'):
    todo = pending(tasks, decoder)
    print(f'{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run')
    if not todo:
        return
    os.makedirs(os.path.dirname(todo[0][0]) or '.', exist_ok=True)
    prepare_luts([code for _, code in todo])          # build each lookup table once, on disk
    args = [(path, code, max_shots, max_errors, shots_step, decoder) for path, code in todo]
    t0 = time.time()
    from tqdm.auto import tqdm
    if workers == 1:
        for path, res in tqdm((_run_one(a) for a in args), total=len(args)):
            _append(path, res)
    else:
        with Pool(workers) as pool:
            for path, res in tqdm(pool.imap(_run_one, args, chunksize=1), total=len(args)):
                _append(path, res)
    print(f'finished in {(time.time() - t0) / 3600:.2f} h')


# --------------------------------------------------------------------------- loading
def collect_table(M_values, r, distances, bases, results_dir, decoder='BetterLUT', ancillas='waiting'):
    """table[M][d][basis] = {p: (rounds, errors, shots)} (rounds sorted)."""
    table = {}
    for M in M_values:
        table[M] = {}
        for d in distances:
            table[M][d] = {}
            for basis in bases:
                by_p = {}
                for rec in load_records(group_file(results_dir, M, r, d, basis)):
                    if rec.decoder != decoder:
                        continue
                    if M > 1 and getattr(rec.code, 'readout_noise_ancillas', None) != ancillas:
                        continue
                    p = float(rec.code.before_round_data_depolarization)
                    # several records of the same task (top-up runs) are independent samples
                    # of the same circuit: add their errors and shots
                    acc = by_p.setdefault(p, {}).setdefault(int(rec.code.rounds), [0, 0])
                    acc[0] += int(rec.errors); acc[1] += int(rec.shots)
                table[M][d][basis] = {}
                for p, by_R in by_p.items():
                    Rs = sorted(by_R)
                    table[M][d][basis][p] = (np.array(Rs), np.array([by_R[R][0] for R in Rs]),
                                             np.array([by_R[R][1] for R in Rs]))
    return table


# --------------------------------------------------------------------------- epsilon
def fit_eps(rounds, errors, shots, drop_end_rounds=True):
    """epsilon per cycle with the original fit_fidelity_vs_rounds (unweighted, same bounds).
    drop_end_rounds=True: drop first and last round when there are more than 3 (original)."""
    R = np.asarray(rounds)
    F = 1 - np.asarray(errors, float) / np.asarray(shots, float)
    n0, n1 = (1, -1) if drop_end_rounds else (0, None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # 'covariance could not be estimated' at tiny counts
            popt, _ = fit_fidelity_vs_rounds(R, F, n0=n0, n1=n1)
    except Exception:
        return np.nan
    return float(10 ** popt[-1])


def eps_curves(table_M, distances, bases, drop_end_rounds=True, n_boot=0, seed=0):
    """Basis-averaged epsilon on the p values present for every distance and basis.

    Returns p (sorted), eps[d] (len(p)), eps_boot[d] (n_boot, len(p)) or None, n_err[d]."""
    rng = np.random.default_rng(seed)
    common = None
    for d in distances:
        for b in bases:
            ps = set(table_M[d][b])
            common = ps if common is None else common & ps
    p = np.array(sorted(common or []))
    eps, boot, n_err = {}, {}, {}
    for d in distances:
        vals = np.zeros((len(bases), len(p)))
        bs = np.zeros((n_boot, len(bases), len(p)))
        ne = np.zeros(len(p), dtype=np.int64)
        for ib, b in enumerate(bases):
            for ip, pp in enumerate(p):
                R, e, s = table_M[d][b][pp]
                vals[ib, ip] = fit_eps(R, e, s, drop_end_rounds)
                ne[ip] += int(np.sum(e))
                for k in range(n_boot):
                    bs[k, ib, ip] = fit_eps(R, rng.binomial(s.astype(np.int64), e / s), s, drop_end_rounds)
        eps[d] = vals.mean(axis=0)
        boot[d] = bs.mean(axis=1) if n_boot else None
        n_err[d] = ne
    return p, eps, (boot if n_boot else None), n_err


# --------------------------------------------------------------------------- crossing point (as in plot_results_phenom)
def _result_dict(p, eps, distances):
    """Minimal stand-in for process_data_logical_vs_rounds output, as estimate_crossing_point expects."""
    pe = np.zeros((len(p), 4)); pe[:, 0] = p
    return {'distances': np.array(distances), 'reference_error': 0, 'phys_error_rate': pe,
            'epsilon': np.array([eps[d] for d in distances])[:, None, :, None]}


def crossing_once(p, eps, distances, drop_end_p=True):
    """estimate_crossing_point + collapsed straight-line fit (Figure 3 cell of plot_results_phenom).
    Returns dict(p_c, nu, poly, pL_c, crossings, found, p_fit)."""
    sel = slice(1, -1) if drop_end_p else slice(None)
    p_fit = p[sel]
    e_fit = {d: eps[d][sel] for d in distances}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cr = estimate_crossing_point(_result_dict(p_fit, e_fit, distances), decoder_index=0, basis_index=0,
                                plot=False, plot_collapse=False, verbose=False)
        found = not any('No crossing' in str(x.message) for x in w)
    x_all, y_all = [], []
    for d in cr['distances']:
        x_all.extend(_fss_scaling_variable(cr['p_grid'], cr['p_c'], cr['nu'], d))
        y_all.extend(np.log(np.maximum(cr['pL_by_distance'][d], 1e-300)))
    x_all, y_all = np.array(x_all), np.array(y_all)
    order = np.argsort(x_all)
    poly = np.polyfit(x_all[order], y_all[order], 1)
    return dict(p_c=float(cr['p_c']), nu=float(cr['nu']), poly=poly, pL_c=float(np.exp(np.polyval(poly, 0.0))),
                crossings=cr['crossing_points'], found=found, p_fit=p_fit)


def analyse(table_M, distances, bases, drop_end_rounds=True, drop_end_p=True, n_boot=200, seed=0):
    """epsilon curves, crossing point and bootstrap errors for one M."""
    p, eps, eps_boot, n_err = eps_curves(table_M, distances, bases, drop_end_rounds, n_boot, seed)
    cr = crossing_once(p, eps, distances, drop_end_p)
    pb, found_b = [], 0
    for k in range(n_boot):
        tb = crossing_once(p, {d: eps_boot[d][k] for d in distances}, distances, drop_end_p)
        if tb['found']:
            pb.append(tb['p_c']); found_b += 1
    pb = np.array(pb)
    return dict(p=p, eps=eps, eps_err={d: (np.nanstd(eps_boot[d], axis=0, ddof=1) if n_boot else None) for d in distances},
                n_err=n_err, cross=cr, p_c_err=float(np.std(pb, ddof=1)) if len(pb) > 2 else np.nan,
                p_c_ci68=tuple(np.percentile(pb, [16, 84])) if len(pb) > 2 else (np.nan, np.nan),
                boot_found=f'{found_b}/{n_boot}', drop_end_rounds=drop_end_rounds, drop_end_p=drop_end_p)


def line_curve(res, d, p_fine):
    """Collapsed straight-line model for distance d (the lines drawn in Figure 3)."""
    cr = res['cross']
    return np.exp(np.polyval(cr['poly'], _fss_scaling_variable(p_fine, cr['p_c'], cr['nu'], d)))


def fmt_sig(x, sig=2):
    """Format x to `sig` significant figures in plain notation (as in plot_results_phenom)."""
    if x == 0 or not np.isfinite(x):
        return str(x)
    from decimal import Decimal
    exponent = Decimal(str(x)).adjusted()
    return f'{x:.{max(sig - exponent - 1, 0)}f}'


def fallback_table(M_values, r, distances, bases, results_dir, decoder='HybridLUT', ancillas='waiting'):
    """Fraction of shots decoded by the Tesseract fallback (HybridLUT), per (M, d, p),
    pooled over rounds and bases.  Returns a list of dict rows."""
    rows = []
    for M in M_values:
        for d in distances:
            acc = {}
            for basis in bases:
                for rec in load_records(group_file(results_dir, M, r, d, basis)):
                    if rec.decoder != decoder:
                        continue
                    if M > 1 and getattr(rec.code, 'readout_noise_ancillas', None) != ancillas:
                        continue
                    p = float(rec.code.before_round_data_depolarization)
                    a = acc.setdefault(p, [0, 0, {}])
                    a[0] += int(getattr(rec, 'fallback', 0)); a[1] += int(rec.shots)
                    R = int(rec.code.rounds)
                    b = a[2].setdefault(R, [0, 0]); b[0] += int(getattr(rec, 'fallback', 0)); b[1] += int(rec.shots)
            for p in sorted(acc):
                f, s, byR = acc[p]
                Rmax = max(byR)
                rows.append(dict(M=M, d=d, p_percent=100 * p, fallback_fraction=f / s if s else np.nan,
                                 fallback_fraction_longest_run=byR[Rmax][0] / byR[Rmax][1] if byR[Rmax][1] else np.nan,
                                 longest_run=Rmax))
    return rows
