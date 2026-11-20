"""
Finite-Size Scaling (FSS) crossing-point estimation for QEC.

Drop this into process_data.py (or import it alongside it).

Usage
-----
Call `estimate_crossing_point` with the list of `result` dicts returned by
`process_data_logical_vs_rounds` — one dict per code distance — or with
a single dict that already contains multiple distances.

Example
-------
    results = []
    for data in [data_d3, data_d5, data_d7, data_d9]:
        results.append(process_data_logical_vs_rounds(data, plot=False))

    crossing_result = estimate_crossing_point(results)
    # or, if all distances are in one result dict:
    crossing_result = estimate_crossing_point(result)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Core FSS functions
# ─────────────────────────────────────────────────────────────────────────────

def _fss_scaling_variable(p, p_c, nu, d):
    """
    FSS scaling variable  x = log(p / p_c) * d^(1/nu).

    Using log(p/p_c) rather than (p - p_c) keeps x O(1) when the data
    spans several decades on a log scale, making the collapse well-conditioned.
    The two forms are equivalent near p_c (by l'Hopital) but log(p/p_c)
    is far more numerically stable for log-spaced p grids.
    """
    return np.log(np.maximum(p, 1e-300) / p_c) * d ** (1.0 / nu)


def _collapse_variance(nu, p_c, p_grid, pL_by_distance):
    """
    Measure data-collapse quality for given (p_c, nu).

    Rescales each distance's pL curve onto the common scaling variable x,
    interpolates all curves onto a shared x grid, then returns the mean
    variance of log(pL) across distances at each x point.  Perfect collapse
    -> zero variance.  This objective is well-conditioned because it does
    not require a free polynomial — it just asks whether the curves overlap.
    """
    distances = sorted(pL_by_distance.keys())
    x_by_d    = {d: _fss_scaling_variable(p_grid, p_c, nu, d) for d in distances}
    logpL_by_d = {d: np.log(np.maximum(pL_by_distance[d], 1e-300)) for d in distances}

    # Only evaluate on the x range shared by all distances
    x_min = max(x_by_d[d].min() for d in distances)
    x_max = min(x_by_d[d].max() for d in distances)
    if x_max <= x_min:
        return 1e10

    x_common = np.linspace(x_min, x_max, 80)
    interps = []
    for d in distances:
        f = interp1d(x_by_d[d], logpL_by_d[d], kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        interps.append(f(x_common))

    return float(np.mean(np.var(np.array(interps), axis=0)))


def _estimate_crossing_points(p_grid, pL_by_distance):
    """
    For each adjacent pair of distances, find p where the pL curves cross.
    Returns a list of crossing p values (sub-grid precision via linear interpolation).
    """
    distances = sorted(pL_by_distance.keys())
    crossings = []
    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i + 1]
        diff = pL_by_distance[d1] - pL_by_distance[d2]
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        for idx in sign_changes:
            p0, p1 = p_grid[idx], p_grid[idx + 1]
            d0, d1_ = diff[idx], diff[idx + 1]
            if d1_ != d0:
                p_cross = p0 - d0 * (p1 - p0) / (d1_ - d0)
                crossings.append(float(p_cross))
    return crossings


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def estimate_crossing_point(
    results,
    decoder_index=0,
    basis_index=0,
    nu_bounds=(0.3, 5.0),
    p_c_bounds=None,
    plot=True,
    plot_collapse=True,
    colors=None,
    verbose=True,
):
    """
    Estimate the crossing point using Finite-Size Scaling (FSS).

    The FSS ansatz is::

        pL(p, d) ≈ f( log(p / p_c) · d^(1/ν) )

    where f is the universal scaling function.  The crossing point p_c is
    estimated from the pairwise crossing points of adjacent-distance curves,
    and ν is extracted by minimising the data-collapse variance — the degree
    to which all curves land on a single f when plotted against the scaling
    variable.

    Parameters
    ----------
    results : dict or list of dict
        Either a single result dict from `process_data_logical_vs_rounds`
        (which may contain multiple distances), or a list of such dicts
        (one per distance or per dataset).  Each dict must contain the
        keys: 'epsilon', 'distances', 'phys_error_rate', 'decoders',
        'basis', 'reference_error'.

    decoder_index : int, default 0
        Which decoder to use (index into result['decoders']).

    basis_index : int, default 0
        Which basis to use (index into result['basis']).

    nu_bounds : tuple (float, float), default (0.3, 5.0)
        Search bounds for ν (critical exponent).

    p_c_bounds : tuple (float, float) or None
        Search bounds for p_c.  If None (default), set automatically to
        [p_min/10, p_max*10] so the crossing is never outside the window.

    plot : bool, default True
        Plot log-log pL vs p with the crossing point marked.

    plot_collapse : bool, default True
        Plot the FSS data collapse (log(pL) vs scaling variable x).

    colors : list of str or None
        Matplotlib colors, one per distance.

    verbose : bool, default True
        Print fit results and crossing-point estimates.

    Returns
    -------
    result_dict : dict with keys
        'p_c'           : float  — estimated crossing point (median of crossings)
        'p_c_err'       : float  — std of crossing points (spread estimate)
        'nu'             : float  — estimated critical exponent
        'nu_err'         : float  — uncertainty on ν from curvature of cost
        'crossing_points': list   — raw pairwise crossing p values
        'crossing_mean'  : float  — mean of crossing points
        'distances'      : array  — distances used
        'p_grid'         : array  — physical error rates used
        'pL_by_distance' : dict   — {d: pL array}
        'fit_success'    : bool
    """
    # ── 1. Collect (p, d, pL) from the result dict(s) ───────────────────────
    if isinstance(results, dict):
        results = [results]

    pL_by_distance = {}
    p_grid_sets    = {}

    for res in results:
        distances = res['distances']
        ref  = res['reference_error']
        p_all = res['phys_error_rate'][:, ref]
        sort_idx  = np.argsort(p_all)
        p_sorted  = p_all[sort_idx]

        for i_d, d in enumerate(distances):
            pL = res['epsilon'][i_d, decoder_index, :, basis_index]
            pL_sorted = pL[sort_idx]
            if d in pL_by_distance:
                existing_p = p_grid_sets[d]
                if np.allclose(existing_p, p_sorted):
                    pL_by_distance[d] = (pL_by_distance[d] + pL_sorted) / 2.0
                else:
                    warnings.warn(
                        f"Distance d={d} appears in multiple result dicts with "
                        "different p grids — using the first occurrence only."
                    )
            else:
                pL_by_distance[d] = pL_sorted
                p_grid_sets[d]    = p_sorted

    if len(pL_by_distance) < 2:
        raise ValueError(
            "FSS requires at least 2 code distances. "
            f"Only found: {list(pL_by_distance.keys())}"
        )

    # Common p_grid
    all_p_sets = list(p_grid_sets.values())
    p_grid = all_p_sets[0]
    for p_set in all_p_sets[1:]:
        p_grid = np.intersect1d(p_grid, p_set)

    if len(p_grid) < 3:
        raise ValueError(
            "Fewer than 3 shared physical error rate values across distances. "
            "Run simulations on the same p grid for all distances."
        )

    for d in list(pL_by_distance.keys()):
        mask = np.isin(p_grid_sets[d], p_grid)
        pL_by_distance[d] = pL_by_distance[d][mask]

    distances_sorted = np.array(sorted(pL_by_distance.keys()))

    # Auto-set p_c bounds from data
    if p_c_bounds is None:
        p_c_bounds = (p_grid.min() / 10.0, p_grid.max() * 10.0)

    # ── 2. Step 1 — estimate p_c from pairwise crossing points ─────────────
    crossings = _estimate_crossing_points(p_grid, pL_by_distance)

    if not crossings:
        warnings.warn(
            "No crossing points found between adjacent distance curves. "
            "Check that your p range straddles the crossing point. "
            "Falling back to geometric mean of p_grid as p_c estimate."
        )
        p_c_fit = float(np.sqrt(p_grid.min() * p_grid.max()))
        p_c_err = float('nan')
    else:
        p_c_fit = float(np.median(crossings))
        p_c_err = float(np.std(crossings)) if len(crossings) > 1 else 0.0

    p_c_fit = np.clip(p_c_fit, p_c_bounds[0], p_c_bounds[1])

    # ── 3. Step 2 — estimate ν by minimising data-collapse variance ──────────
    #  Coarse grid scan first, then local refinement
    nu_coarse = np.linspace(nu_bounds[0], nu_bounds[1], 120)
    costs_coarse = np.array([
        _collapse_variance(nu, p_c_fit, p_grid, pL_by_distance)
        for nu in nu_coarse
    ])
    nu_init = nu_coarse[np.argmin(costs_coarse)]

    opt = minimize(
        lambda nu_arr: _collapse_variance(nu_arr[0], p_c_fit, p_grid, pL_by_distance),
        x0=[nu_init],
        method='L-BFGS-B',
        bounds=[(nu_bounds[0], nu_bounds[1])],
        options={'maxiter': 2000, 'ftol': 1e-14},
    )
    nu_fit = float(opt.x[0])
    fit_success = bool(opt.success)

    # Uncertainty on ν from curvature of collapse-variance cost near minimum
    try:
        h = nu_fit * 1e-3
        f0 = _collapse_variance(nu_fit,     p_c_fit, p_grid, pL_by_distance)
        fp = _collapse_variance(nu_fit + h,  p_c_fit, p_grid, pL_by_distance)
        fm = _collapse_variance(nu_fit - h,  p_c_fit, p_grid, pL_by_distance)
        d2f = (fp - 2*f0 + fm) / h**2
        nu_err = float(1.0 / np.sqrt(max(d2f, 1e-30)))
    except Exception:
        nu_err = float('nan')

    # ── 4. Verbose output ────────────────────────────────────────────────────
    if verbose:
        print("=" * 55)
        print("  FSS Crossing-Point Estimation Results")
        print("=" * 55)
        print(f"  Distances used   : {distances_sorted.tolist()}")
        print(f"  Decoder index    : {decoder_index}")
        print(f"  Basis index      : {basis_index}")
        print(f"  Fit success      : {fit_success}")
        print(f"  Collapse cost    : {opt.fun:.4e}")
        print("-" * 55)
        if crossings:
            print(f"  Crossing points  : {[f'{c:.3e}' for c in crossings]}")
            print(f"  Crossing mean    : {np.mean(crossings):.3e}")
        print(f"  p_c (crossings) : {p_c_fit:.3e}  ± {p_c_err:.3e}")
        print(f"  ν   (collapse)   : {nu_fit:.4f}  ± {nu_err:.4f}")
        print("=" * 55)

    # ── 5. Build a smooth collapsed curve for plotting ───────────────────────
    # Fit a cubic polynomial to the collapsed (x, log_pL) data for the overlay
    x_all_pts, logpL_all_pts = [], []
    for d in distances_sorted:
        x_all_pts.extend(_fss_scaling_variable(p_grid, p_c_fit, nu_fit, d))
        logpL_all_pts.extend(np.log(np.maximum(pL_by_distance[d], 1e-300)))
    x_all_pts    = np.array(x_all_pts)
    logpL_all_pts = np.array(logpL_all_pts)
    sort_idx = np.argsort(x_all_pts)
    poly_coeffs = np.polyfit(x_all_pts[sort_idx], logpL_all_pts[sort_idx], 1)[::-1]

    # ── 6. Plot: log-log pL vs p ─────────────────────────────────────────────
    if colors is None:
        colors = ['C' + str(i) for i in range(10)] * 10
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        markers = ['o', 's', '^', 'D', 'v', 'p'] * 5
        for i, d in enumerate(distances_sorted):
            ax.plot(
                p_grid, pL_by_distance[d],
                markers[i], color=colors[i],
                label=f'd={d}', markersize=7,
            )
            # Overlay FSS model curve: exp(polynomial(x(p)))
            p_fine = np.geomspace(p_grid.min(), p_grid.max(), 300)
            x_fine = _fss_scaling_variable(p_fine, p_c_fit, nu_fit, d)
            pL_fine = np.exp(np.polyval(poly_coeffs[::-1], x_fine))
            ax.plot(p_fine, pL_fine, '-', color=colors[i], alpha=0.6)
        ax.axvline(p_c_fit, color='k', linestyle='--', linewidth=1.5,
                   label=f'$p_c$ = {p_c_fit:.3e}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate $p$')
        ax.set_ylabel('Logical error rate $p_L$')
        ax.set_title(
            f'FSS Crossing-Point Estimate\n'
            f'$p_c={p_c_fit:.3e}\\pm{p_c_err:.1e}$,  '
            f'$\\nu={nu_fit:.3f}\\pm{nu_err:.3f}$'
        )
        ax.legend(loc='upper left')
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    # ── 7. Plot: FSS data collapse ───────────────────────────────────────────
    # x-axis is the scaling variable (spans negative to positive, stays linear)
    # y-axis is log(pL)
    if plot_collapse:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        markers = ['o', 's', '^', 'D', 'v', 'p'] * 5
        x_all_plot = []
        for i, d in enumerate(distances_sorted):
            x_scaled = _fss_scaling_variable(p_grid, p_c_fit, nu_fit, d)
            ax2.semilogy(
                x_scaled, pL_by_distance[d],
                markers[i], color=colors[i],
                label=f'd={d}', markersize=7,
            )
            x_all_plot.extend(x_scaled.tolist())
        # Overlay the fitted collapse curve
        x_range = np.linspace(min(x_all_plot), max(x_all_plot), 400)
        pL_range = np.exp(np.polyval(poly_coeffs[::-1], x_range))
        mask = pL_range > 0
        ax2.semilogy(x_range[mask], pL_range[mask],
                     'k-', linewidth=2, label='$f(x)$ fit', zorder=10)
        ax2.axvline(0, color='gray', linestyle=':', linewidth=1,
                    label='$x=0$ ($p=p_c$)')
        ax2.set_xlabel(r'Scaling variable $x = \log(p/p_c)\,d^{1/\nu}$')
        ax2.set_ylabel('Logical error rate $p_L$')
        ax2.set_title(
            'FSS Data Collapse\n'
            f'$p_c={p_c_fit:.3e}$,  $\\nu={nu_fit:.3f}$'
        )
        ax2.legend(loc='upper left')
        ax2.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
        ax2.grid(True, which='minor', axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return {
        'p_c':            p_c_fit,
        'p_c_err':        p_c_err,
        'nu':              nu_fit,
        'nu_err':          nu_err,
        'crossing_points': crossings,
        'crossing_mean':   float(np.mean(crossings)) if crossings else float('nan'),
        'distances':       distances_sorted,
        'p_grid':          p_grid,
        'pL_by_distance':  pL_by_distance,
        'fit_success':     fit_success,
    }
