"""
Monte-Carlo sampling and decoding for the globally controlled ring codes.

All decoders see the SAME shots: detection events and observable flips are sampled
once per batch with Stim's detector sampler and handed to every decoder in
`decoders`.

Decoders
--------
'BetterLUT'   lookup table with memory (Ref. [33]) on the detection events.
'LUT'         same, without the memory preprocessing.
'BetterLUT2'  POST-SELECTED: runs two LUT variants and keeps only shots on which
              they agree.  Rejected shots are stored in TaskStats.discarded and are
              excluded from TaskStats.shots.  Report the post-selection if used.
'HybridLUT'   BetterLUT, plus a fallback: shots on which the two temporal preprocessing
              rules of the lookup table (look one vs two cycles ahead) disagree are
              decoded with Tesseract (search-based, on the circuit's detector error
              model).  Every shot gets a correction (no post-selection); the number of
              fallback shots is stored in TaskStats.fallback.  Needs tesseract-decoder.
'no'          no correction (raw observable flips).
'pymatching', 'beliefmatching'  only for graph-like error models (imported on use).

Lookup tables depend only on the code (name and distance), not on the basis, p or
the number of rounds.  Building the d = 9 table takes ~90 s and ~3.6 GB of memory, so:
* with LUT_CACHE_DIR set (see set_lut_cache_dir / prepare_luts), each table is built
  once, saved as .npy, and every worker memory-maps the same file (the operating
  system shares one copy between processes);
* otherwise each worker builds the table once and keeps the most recent one.
"""
from multiprocessing import Pool
from dataclasses import dataclass, field
from typing import Union
import pickle
import time

import numpy as np
import stim
from tqdm import tqdm

import istarmap  # noqa: F401  (registers Pool.istarmap)
from QECcode import QECcode
from LookupTableDecoder import LookupTableDecoder
from process_data import task_key

LUT_DECODERS = ('LUT', 'BetterLUT', 'BetterLUT2', 'HybridLUT')
TESSERACT_OPTIONS = dict(det_beam=12, beam_climbing=True)   # fallback decoder of 'HybridLUT'
KNOWN_DECODERS = ('no', 'pymatching', 'beliefmatching') + LUT_DECODERS


@dataclass
class TaskStats:
    shots: int          # shots used for the error rate (accepted shots for BetterLUT2)
    errors: int         # logical failures among those shots
    decoder: str
    code: Union[QECcode, str]
    hash_str: str       # str(task_key(code, decoder)); kept for backward compatibility
    discarded: int = 0  # shots rejected by post-selection (BetterLUT2 only)
    fallback: int = 0   # shots decoded by the Tesseract fallback (HybridLUT only)
    seconds: float = 0. # wall time of the task
    key: tuple = field(default=None)


# --------------------------------------------------------------------------- lookup-table cache
import os
_LUT_CACHE = {'key': None, 'lut': None}
LUT_CACHE_DIR = os.environ.get('LUT_CACHE_DIR')     # inherited by worker processes


def set_lut_cache_dir(path):
    """Directory for saved lookup tables (also exported so worker processes see it)."""
    global LUT_CACHE_DIR
    LUT_CACHE_DIR = path
    os.environ['LUT_CACHE_DIR'] = path
    os.makedirs(path, exist_ok=True)


def _lut_file(code):
    return os.path.join(LUT_CACHE_DIR, f'lut_{code.name}_d{int(code.distance)}.npy')


def prepare_luts(codes):
    """Build and save the table of every distinct code once (call in the main process)."""
    if LUT_CACHE_DIR is None:
        raise ValueError('call set_lut_cache_dir(path) first')
    done = set()
    for code in codes:
        key = (code.name, int(code.distance))
        if key in done:
            continue
        done.add(key)
        path = _lut_file(code)
        if not os.path.exists(path):
            t0 = time.time()
            table = LookupTableDecoder(code).lookuptable
            np.save(path + '.tmp.npy', table)
            os.replace(path + '.tmp.npy', path)
            print(f'built lookup table for {code.name} d={code.distance} in {time.time() - t0:.0f} s -> {path}')


def get_lut(code):
    """Lookup table for this code; reused while consecutive tasks share (name, distance, basis)."""
    key = (code.name, int(code.distance), code.basis)
    if _LUT_CACHE['key'] != key:
        _LUT_CACHE['lut'] = None                 # free the old table before loading the new one
        if LUT_CACHE_DIR is not None and os.path.exists(_lut_file(code)):
            table = np.load(_lut_file(code), mmap_mode='r')
            _LUT_CACHE['lut'] = LookupTableDecoder(code, lookuptable=table)
        else:
            _LUT_CACHE['lut'] = LookupTableDecoder(code)
        _LUT_CACHE['key'] = key
    return _LUT_CACHE['lut']


# --------------------------------------------------------------------------- detector layout for the LUT
def detectors_for_lut(circuit: stim.Circuit):
    """Index array that reorders detection events into (cycle, stabilizer) order.

    Returns idx with shape (cycles, n_syndromes) so that dets[:, idx] has shape
    (shots, cycles, n_syndromes).  Uses the detector coordinates
    (stabilizer_index, cycle_index) written by GlobalRingCode."""
    coords = circuit.get_detector_coordinates()
    stabs = sorted({int(c[0]) for c in coords.values()})
    cycles = sorted({int(c[1]) for c in coords.values()})
    idx = np.full((len(cycles), len(stabs)), -1, dtype=np.int64)
    s_pos = {s: j for j, s in enumerate(stabs)}
    c_pos = {c: j for j, c in enumerate(cycles)}
    for det, c in coords.items():
        idx[c_pos[int(c[1])], s_pos[int(c[0])]] = det
    if np.any(idx < 0):
        raise ValueError('detector layout is not a full (cycle x stabilizer) grid; '
                         'was the circuit built by GlobalRingCode?')
    return idx


def lut_input(dets: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(shots, cycles+1, n_syndromes) detection events with a leading all-zero cycle,
    the format LookupTableDecoder.decode_batch(detector=True) expects."""
    x = dets[:, idx].astype(np.int64)
    return np.concatenate([np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.int64), x], axis=1)


# --------------------------------------------------------------------------- main loop
def collect(tasks: list, max_shots: int, max_errors: int, decoders=None, workers=1, shots_step=10000):
    """Sample every task until max_shots, or until EVERY decoder has max_errors failures.
    Returns a flat list of TaskStats (one per task and decoder)."""
    if decoders is None:
        decoders = ['no']
    for dec in decoders:
        if dec not in KNOWN_DECODERS:
            raise ValueError(f'unknown decoder {dec!r}; choose from {KNOWN_DECODERS}')
    shots_step = min(max_shots, shots_step)
    assert workers > 0 and type(workers) == int
    assert max_errors > 0 and max_shots > 0
    iterable = [(task, max_shots, max_errors, decoders, shots_step) for task in tasks]
    if workers == 1:
        data = [single_run(*args) for args in tqdm(iterable, total=len(tasks))]
    else:
        with Pool(workers) as pool:
            data = list(tqdm(pool.istarmap(single_run, iterable), total=len(tasks)))
    return [x for data_ in data for x in data_]


def single_run(code, max_shots, max_errors, decoders, shots_step):
    t_start = time.time()
    circuit = code if isinstance(code, stim.Circuit) else code.get_circuit()
    sampler = circuit.compile_detector_sampler()

    errors = np.zeros(len(decoders), dtype=np.int64)
    discarded = np.zeros(len(decoders), dtype=np.int64)
    fallback = np.zeros(len(decoders), dtype=np.int64)
    tess = None                                   # built only if a fallback is needed

    lut = idx = None
    if any(d in LUT_DECODERS for d in decoders):
        lut = get_lut(code)
        idx = detectors_for_lut(circuit)
    matching = None
    if 'pymatching' in decoders:
        import pymatching
        matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    bm = None
    if 'beliefmatching' in decoders:
        import beliefmatching
        bm = beliefmatching.BeliefMatching(circuit, max_bp_iters=20)

    shots = 0
    while shots < max_shots and errors.min() < max_errors:
        step = int(min(shots_step, max_shots - shots))
        dets, obs = sampler.sample(shots=step, separate_observables=True)
        obs = obs.astype(np.uint8)
        x_lut = lut_input(dets, idx) if lut is not None else None
        for i, decoder in enumerate(decoders):
            if decoder == 'no':
                errors[i] += np.sum(np.any(obs, axis=1))
            elif decoder == 'pymatching':
                errors[i] += np.sum(np.any(matching.decode_batch(dets) != obs, axis=1))
            elif decoder == 'beliefmatching':
                errors[i] += np.sum(np.any(bm.decode_batch(dets) != obs, axis=1))
            else:   # lookup-table family; single logical qubit -> observable 0
                o = obs[:, 0].astype(np.int64)
                dec = lut.decode_batch(x_lut, better_preproc=(decoder != 'LUT'), detector=True)
                if decoder == 'BetterLUT2':
                    dec2 = lut.decode_batch(x_lut, better_preproc=False, better_preproc2=True,
                                            detector=True, correct_history=False)
                    agree = dec == dec2
                    errors[i] += np.sum(((o + dec) % 2) * agree)
                    discarded[i] += np.sum(~agree)
                elif decoder == 'HybridLUT':
                    dec2 = lut.decode_batch(x_lut, better_preproc=False, better_preproc2=True,
                                            detector=True, correct_history=False)
                    dis = np.nonzero(dec != dec2)[0]
                    if len(dis):
                        if tess is None:
                            from tesseract_decoder import tesseract
                            tess = tesseract.TesseractConfig(dem=circuit.detector_error_model(),
                                                             **TESSERACT_OPTIONS).compile_decoder()
                        dec = dec.copy()
                        dec[dis] = tess.decode_batch(dets[dis])[:, 0].astype(dec.dtype)
                        fallback[i] += len(dis)
                    errors[i] += np.sum((o + dec) % 2)
                else:
                    errors[i] += np.sum((o + dec) % 2)
        shots += step

    seconds = time.time() - t_start
    out = []
    for i, decoder in enumerate(decoders):
        key = task_key(code, decoder) if not isinstance(code, stim.Circuit) else None
        out.append(TaskStats(shots=int(shots - discarded[i]), errors=int(errors[i]), decoder=decoder,
                             code=code, hash_str=str(key), discarded=int(discarded[i]),
                             fallback=int(fallback[i]), seconds=float(seconds), key=key))
    return out


def save_pkl(file, obj):
    with open(file, 'wb') as fileObj:
        pickle.dump(obj, fileObj)


def load_pkl(file):
    with open(file, 'rb') as fileObj:
        return pickle.load(fileObj)
