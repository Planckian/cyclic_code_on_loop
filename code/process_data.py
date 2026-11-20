import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Identifying a simulation task
# ---------------------------------------------------------------------------
# Results used to be matched with a string hash that left out k, the
# measurement flip probability and other settings, so runs that differed only
# in those were silently confused.  task_key() includes every setting that
# changes the circuit.  getattr defaults keep old pickles loadable: runs made
# before readout_noise_ratio/readout_noise_ancillas existed had data-only
# readout noise at r = 1, i.e. (1.0, 'none').

def _f(x):
    return float(x) if x is not None else 0.0


def code_config(code):
    """Settings that are shared by one curve (everything except p, basis and rounds)."""
    k = getattr(code, 'k', None)
    return (code.name, int(code.distance), None if k is None else int(k),
            float(getattr(code, 'readout_noise_ratio', 1.0)),
            str(getattr(code, 'readout_noise_ancillas', 'none')),
            tuple(getattr(code, 'native_2q_gate_set', ['iSWAP'])),
            _f(getattr(code, 'after_reset_flip_probability', 0.)),
            int(getattr(code, 'rounds_before', 1)), int(getattr(code, 'rounds_after', 1)))


def phys_rates(code):
    """(DD, Q1, Q2, M) = data depolarization, 1q gate, 2q gate, measurement flip."""
    return (_f(code.before_round_data_depolarization), _f(code.after_clifford_depolarization_Q1),
            _f(code.after_clifford_depolarization_Q2), _f(code.before_measure_flip_probability))


def task_key(code, decoder):
    return code_config(code) + phys_rates(code) + (code.basis, int(code.rounds), decoder)


def func_fidelity_vs_rounds(t, t0, epsilon10):
    return 1 / 2 * (1 + (1 - 2 * 10**epsilon10) ** (t) * t0)


def fit_fidelity_vs_rounds(t_axis, fidelity, n0=1, n1=-1):
    bounds = ([0.5, -10], [1.2, 1.])
    if len(t_axis) > 3:
        popt, pcov = curve_fit(func_fidelity_vs_rounds, t_axis[n0:n1], fidelity[n0:n1], bounds=bounds)
    else:
        popt, pcov = curve_fit(func_fidelity_vs_rounds, t_axis, fidelity, bounds=bounds)
    return popt, pcov


def logical_error_fit(plist, epsilon, n0=0, n1=-1):
    x = np.log10(plist[n0:n1])
    y = np.log10(epsilon[n0:n1])
    popt, pcov = curve_fit(line, x, y)
    return popt, pcov

def line(x, a, b):
    return a * x + b

def line_fit(x, a, b):
    return 10**(a * np.log10(x) + b)


def get_unique_rounds(data):
    rounds = []
    for data_ in data:
        if data_.code.rounds not in rounds:
            rounds.append(data_.code.rounds)
    return np.asarray(rounds)


def get_unique_distances(data):
    distances = []
    for data_ in data:
        if data_.code.distance not in distances:
            distances.append(data_.code.distance)
    return np.asarray(distances)


def get_unique_basis(data):
    basis = []
    for data_ in data:
        if data_.code.basis not in basis:
            basis.append(data_.code.basis)
    return np.asarray(basis)


def get_unique_decoder(data, inactive_decoders):
    decoders = []
    for data_ in data:
        if data_.decoder not in decoders and data_.decoder not in inactive_decoders:
            decoders.append(data_.decoder)
    return np.asarray(decoders)


def get_unique_phys_error_rates(data):
    phys_error_rate = []
    errors_str = ['DD', 'Q1', 'Q2', 'M']
    for data_ in data:
        error_ = phys_rates(data_.code)
        if error_ not in phys_error_rate:
            phys_error_rate.append(error_)
    phys_error_rate = np.asarray(phys_error_rate)

    for i in range(len(errors_str)):
        if np.sum(phys_error_rate[:, i]) < 10 ** (-6):
            errors_str[i] = '0*' + errors_str[i]
    reference_error = 0 if phys_error_rate[-1, 0] > 1e-6 else 2
    #reference_error = 0
    #for i in range(len(errors_str)):
    #    if errors_str[i][0] == '0':
    #        reference_error += 1
    #    else:
    #        break
    for i in range(reference_error + 1, len(errors_str), 1):
        if errors_str[i][0] != '0':
            errors_str[i] = errors_str[i] + '=' + str(phys_error_rate[-1, i] / phys_error_rate[-1, reference_error])[
                                                  :5] + \
                            '*' + errors_str[reference_error]
    if reference_error >= len(errors_str):
        reference_error = 0
    return phys_error_rate, errors_str, reference_error


def process_data_logical_vs_rounds(data, inactive_decoders=[], plot=True, colors=None):
    if colors == None:
        colors: list[str] = ['C' + str(i) for i in range(10)] * 10
    markers = ['o', 'v', '*', 's', 'p', 'D'] * 10

    rounds = get_unique_rounds(data)
    distances = get_unique_distances(data)
    basis = get_unique_basis(data)
    decoders = get_unique_decoder(data, inactive_decoders)
    phys_error_rate, errors_str, reference_error = get_unique_phys_error_rates(data)
    fidelity = np.zeros((len(distances), len(decoders), len(phys_error_rate), len(basis), len(rounds)))
    epsilon = np.zeros((len(distances), len(decoders), len(phys_error_rate), len(basis)))
    epsilon_error = np.zeros((len(distances), len(decoders), len(phys_error_rate), len(basis)))
    # one curve per distance: refuse to mix k, r, gate set, ... within a data list
    configs = {}
    for data_ in data:
        configs.setdefault(int(data_.code.distance), set()).add(code_config(data_.code))
    for d_, cfg in configs.items():
        if len(cfg) > 1:
            raise ValueError(f'data for d={d_} mixes {len(cfg)} different settings (k, r, gate set, ...); '
                             f'split it into separate lists: {sorted(cfg, key=str)}')
    lookup = {}
    for data_ in data:
        key = (int(data_.code.distance), data_.decoder, phys_rates(data_.code), data_.code.basis, int(data_.code.rounds))
        if key in lookup:
            raise ValueError(f'duplicate result for {key}; combine or remove duplicates before processing')
        lookup[key] = data_
    discarded_fraction = np.zeros((len(distances), len(decoders), len(phys_error_rate), len(basis), len(rounds)))

    for i in range(len(distances)):
        for j in range(len(decoders)):
            for k in range(len(phys_error_rate)):
                for b in range(len(basis)):
                    for r in range(len(rounds)):
                        key = (int(distances[i]), decoders[j], tuple(phys_error_rate[k]), basis[b], int(rounds[r]))
                        if key not in lookup:
                            raise KeyError(f'no result for d, decoder, (DD,Q1,Q2,M), basis, rounds = {key}')
                        entry = lookup[key]
                        fidelity[i, j, k, b, r] = 1 - entry.errors / entry.shots
                        discarded = getattr(entry, 'discarded', 0)
                        discarded_fraction[i, j, k, b, r] = discarded / (entry.shots + discarded)
                    if len(rounds) == 1:
                        popt, pcov = fit_fidelity_vs_rounds([0] + rounds.tolist(), [1] + fidelity[i, j, k, b].tolist())
                    else:
                        popt, pcov = fit_fidelity_vs_rounds(rounds, fidelity[i, j, k, b])
                    epsilon[i, j, k, b] = 10**popt[-1]
                    epsilon_error[i, j, k, b] = np.sqrt(pcov[-1,-1])

    if plot:
        fig, ax = plt.subplots(len(distances), len(decoders), figsize=(len(decoders) * 7.8, len(distances) * 3.9),
                               sharex=True, sharey=False)
        for i in range(len(distances)):
            for j in range(len(decoders)):
                if len(distances) == 1 and len(decoders) == 1:
                    axs = ax
                elif len(distances) == 1:
                    axs = ax[j]
                elif len(decoders) == 1:
                    axs = ax[i]
                else:
                    axs = ax[i, j]
                for k in range(len(phys_error_rate)):
                    for b in range(len(basis)):
                        if len(rounds) == 1:
                            popt, pcov = fit_fidelity_vs_rounds([0] + rounds.tolist(),
                                                                [1] + fidelity[i, j, k, b].tolist())
                        else:
                            popt, pcov = fit_fidelity_vs_rounds(rounds, fidelity[i, j, k, b])
                        if basis[b] == "Z":
                            mrk = markers[0]
                        elif basis[b] == "X":
                            mrk = markers[1]
                        elif basis[b] == "Y":
                            mrk = markers[2]
                        axs.plot(rounds, fidelity[i, j, k, b], mrk, color=colors[i],
                                 label=basis[b] + ' ' + errors_str[reference_error] + ' = ' + \
                                       f'{phys_error_rate[k, reference_error] * 100:.3g}' + '%' + \
                                       r' $\epsilon=$' + f'{10**popt[1] * 100:.3g}' + '%')
                        axs.plot(np.arange(max(rounds)+1), func_fidelity_vs_rounds(np.arange(max(rounds)+1), *popt), color=colors[i])
                axs.set_title('distance=' + str(distances[i]) + ' ' + decoders[j] + '\n')
                axs.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                axs.grid('--')
                axs.set_xlabel('Rounds')
                axs.set_ylabel('Fidelity')
        plt.suptitle(data[0].code.name + ' ' + ' '.join(errors_str), fontsize='x-large')
        plt.tight_layout()
        plt.show()

    result = {
        'epsilon': epsilon,
        'epsilon_error_log10': epsilon_error,
        'fidelity': fidelity,
        'distances': distances,
        'rounds': rounds,
        'decoders': decoders,
        'phys_error_rate': phys_error_rate,
        'basis': basis,
        'errors_str': errors_str,
        'reference_error': reference_error,
        'discarded_fraction': discarded_fraction,   # nonzero only for the post-selected BetterLUT2
        'config': {d_: next(iter(c)) for d_, c in configs.items()},
    }

    return result


def process_data_logical_vs_physical(data_list, inactive_decoders=[], plot=True, colors=None, fit=True,
                                     basis_mean=False, logscale=True, xlim=None, ylim=None, label_set=None):
    if colors == None:
        colors: list[str] = ['C' + str(i) for i in range(10)] * 10
    markers = ['o', 'v', '*', 's', 'p', 'D'] * 10
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    suptitle = []
    color_cnt = 0
    for data_i, data in enumerate(data_list):
        result = process_data_logical_vs_rounds(data, inactive_decoders, plot=False)
        suptitle.append(data[0].code.name + ' ' + ' '.join(result['errors_str']))
        for i in range(len(result['distances'])):
            for j in range(len(result['decoders'])):
                x = result['phys_error_rate'][:, result['reference_error']]
                if basis_mean:
                    # average epsilon over the basis axis (last axis), keeping (phys_error_rate,) shape
                    y = np.mean(result['epsilon'][i, j, :, :], axis=-1)
                    if label_set is not None:
                        d, c = label_set[color_cnt]
                        label = r'$d=$' + str(d) + r', $k=$' + str(c)
                    else:
                        label = r'$d=$' + str(result['distances'][i])
                    if len(result['decoders']) > 1:
                        label = label + ' ' + result['decoders'][j]
                    ax.plot(x, y, markers[i], color=colors[color_cnt], markersize=5, label=label)
                    if fit:
                        popt, pcov = logical_error_fit(x, y, n0=1, n1=-1)
                        ax.plot(x, line_fit(x, *popt), color=colors[color_cnt], label=f'slope {popt[0]:.3g}')
                    color_cnt += 1
                else:
                    for b in range(len(result['basis'])):
                        y = result['epsilon'][i, j, :, b]
                        if label_set is not None:
                            d, c = label_set[color_cnt]
                            label = r'$d=$' + str(d) + r', $k=$' + str(c)
                        else:
                            label = r'$d=$' + str(result['distances'][i])
                        if len(result['decoders']) > 1:
                            label = label + ' ' + result['decoders'][j]
                        ax.plot(x, y, markers[i], color=colors[color_cnt], markersize=5, label=label)
                        if fit:
                            popt, pcov = logical_error_fit(x, y, n0=1, n1=-1)
                            ax.plot(x, line_fit(x, *popt), color=colors[color_cnt], label=f'slope {popt[0]:.3g}')
                        color_cnt += 1
                        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # plot the x=y line over the overlapping range of the two axes
    lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='k', linestyle='--', label='threshold')
    ax.set_xlabel(r'Physical error',fontsize=14)
    ax.set_ylabel(r'Logical error',fontsize=14)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.minorticks_on()
    ax.minorticks_on()
    ax.grid(which='major', axis='both')
    ax.grid(which='minor', axis='x', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='x', which='minor', labelsize=12)
    plt.show()
    
    
def process_data_logical_vs_physical_asym(data_list, inactive_decoders=[], plot=True, colors=None, fit=True,
                                     basis_mean=False, logscale=True, xlim=None, ylim=None, label_set=None):
    if colors == None:
        colors: list[str] = ['C' + str(i) for i in range(10)] * 10
    markers = ['o', 'v', '*', 's', 'p', 'D'] * 10
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    suptitle = []
    color_cnt = 0
    for data_i, data in enumerate(data_list):
        result = process_data_logical_vs_rounds(data, inactive_decoders, plot=False)
        suptitle.append(data[0].code.name + ' ' + ' '.join(result['errors_str']))
        for i in range(len(result['distances'])):
            for j in range(len(result['decoders'])):
                for b in range(len(result['basis'])):
                    x = 6.8*result['phys_error_rate'][:, result['reference_error']]
                    y = result['epsilon'][i, j, :, b]
                    if label_set is not None:
                        d, c = label_set[color_cnt]
                        label = r'$d=$' + str(d) + r', $k=$' + str(c)
                    else:
                        label = r'$d=$' + str(result['distances'][i])
                    if len(result['decoders']) > 1:
                        label = label + ' ' + result['decoders'][j]
                    ax.plot(x, y, markers[i], color=colors[color_cnt], markersize=5, label=label)
                    if fit:
                        popt, pcov = logical_error_fit(x, y, n0=1, n1=-1)
                        ax.plot(x, line_fit(x, *popt), color=colors[color_cnt], label=f'slope {popt[0]:.3g}')
                    color_cnt += 1
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # plot the x=y line over the overlapping range of the two axes
    lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='k', linestyle='--', label='threshold')
    ax.set_xlabel(r'Physical error',fontsize=16)
    ax.set_ylabel(r'Logical error',fontsize=16)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.minorticks_on()
    ax.minorticks_on()
    ax.grid(which='major', axis='both')
    ax.grid(which='minor', axis='x', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()