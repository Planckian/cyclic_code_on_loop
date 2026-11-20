# Quantum error correction with global control: code and data

Simulation code and data for Fig. 3 of

> R. Menta, L. Bassman Oftelie, A. Abedi, F. Cioni, M. Polini, S. Lloyd, F. Caravelli and V. Giovannetti,
> *Quantum error correction with global control* (2026).

Fig. 3 shows the logical error rate per syndrome-extraction (SE) cycle of the cyclic codes of distance
d = 5, 7, 9 on the globally controlled ring, for M = 1, 2, 3 rounds of serial syndrome readout, under a
phenomenological noise model. The crossing points p<sub>c</sub><sup>(M)</sup> are about 0.77 %, 0.47 % and
0.33 %.

## Contents

```
data/
  figure3_logical_errors.csv     logical-error counts for every simulated task (the raw data of Fig. 3)
  figure3_points.csv             basis-averaged logical error rate per cycle: the points of Fig. 3
  figure3_crossing_points.csv    crossing points p_c and the collapsed-line fits: the stars and lines of Fig. 3
code/
  QECcode.py                     base class for stim memory circuits
  GlobalRingCode.py              cyclic code on the ring: global SE circuit, serial readout, noise model
  ChainCodeGlobal.py             d = 5 code
  LinearChainCodeGlobal.py       d = 7, 9 codes
  LookupTableDecoder.py          lookup-table decoder with memory (Simakov and Besedin, PRA 111, 012444)
  collectorGlobal.py             sampling and decoding (HybridLUT: lookup table + Tesseract fallback)
  lut_pipeline.py                running the tasks, logical-error-rate fits, crossing points, bootstrap
  fss_crossing.py               adjacent-distance crossings and finite-size collapse
  process_data.py                fidelity-vs-rounds fit
  istarmap.py                    multiprocessing helper
  figure3_lut.ipynb              notebook used to run the simulations and draw Fig. 3
  export_csv.py                  writes data/figure3_logical_errors.csv from the raw result files
  reproduce_figure3.py           reproduces Fig. 3 and the crossing points from the CSV data
  lut_results/lookup_tables/     precomputed lookup tables for d = 5 and d = 7
```

## Data

### `figure3_logical_errors.csv`

One row per simulated task, with the counts of all runs of that task added together.

| column | meaning |
| --- | --- |
| `M` | number of serial readout rounds per SE cycle |
| `d` | code distance (n = 4d − 7 data qubits, N = 2n qubits on the ring) |
| `measurement_sites` | number of local measurement sites ℓ (smallest ℓ giving M rounds, Table 2) |
| `basis` | logical basis prepared and measured (X, Y or Z) |
| `p` | physical error rate (total depolarizing probability; measurement flip probability q = p) |
| `rounds` | number of noisy SE cycles R |
| `shots` | number of shots |
| `logical_errors` | number of shots with a logical error |
| `fallback_shots` | number of shots decoded by the Tesseract fallback instead of the lookup table |
| `in_figure3` | `False` for the p values that were simulated but are not shown in Fig. 3 (the three highest p for M = 1 and the highest for M = 2, 3) |

Each task was sampled in batches of 5000 shots until at least 1000 logical errors or 5 × 10<sup>5</sup> shots.

### `figure3_points.csv`

`logical_error_rate_per_cycle` is the logical error rate ε per SE cycle, obtained by fitting
F(R) = ½[1 + F<sub>0</sub>(1 − 2ε)<sup>R</sup>] to all R and averaging ε over the three bases.
`bootstrap_std` is its standard deviation over 200 bootstrap resamples of every binomial count, and
`logical_errors_total` the number of logical errors behind the point, summed over bases and R.

### `figure3_crossing_points.csv`

For each M: the crossing point `p_c` (median of the crossings of adjacent distances (5, 7) and (7, 9),
listed in `adjacent_distance_crossings`), its bootstrap standard deviation and 68 % interval, the exponent
`nu` and the straight-line fit log ε = `line_a` + `line_b` · x with x = log(p/p<sub>c</sub>) d<sup>1/ν</sup>.
The stars of Fig. 3 are at (p<sub>c</sub>, e<sup>a</sup>).

## Reproducing Fig. 3 from the data

```bash
pip install -r requirements.txt
cd code
python reproduce_figure3.py
```

This rebuilds `data/figure3_points.csv` and `data/figure3_crossing_points.csv` and writes `code/figure3_XYZ.pdf`
(about 2 minutes, most of it bootstrap).

## Rerunning the simulations

`code/figure3_lut.ipynb` runs the simulations (Sections 1 and 2) and draws Fig. 3 (Section 3). It writes the raw
results to `code/lut_results/M<M>_r1_d<d>_<basis>.pkl`, which `export_csv.py` turns into the CSV file. The raw
result files are not included in this repository.

The lookup table of the d = 9 code (537 MB) is not included either. It is built automatically on the first d = 9
run and saved in `code/lut_results/lookup_tables/`.

The simulations need `tesseract-decoder` in addition to the packages used for the analysis:

```bash
pip install -r requirements.txt tesseract-decoder
```

### Noise model

Per SE cycle: a depolarizing channel of strength p on every data qubit; a bit flip of probability p before each
syndrome measurement; and, after each readout round except the last, a depolarizing channel of strength p on the
data qubits and on the syndrome qubits not yet measured in that cycle. Syndrome qubits that have already been
measured are reset and noiseless until the next cycle. Gates are noiseless. Each memory experiment has one
noiseless SE cycle before and one after the R noisy cycles.

### Decoder

Each shot is decoded with the lookup table with memory, using two variants of its temporal preprocessing. When
they agree, their correction is used; when they disagree, the shot is decoded with Tesseract (beam width 12, with
beam climbing) on the circuit's detector error model. No shots are discarded.

## License

The code is released under the MIT License ([LICENSE](LICENSE)). The data in `data/` are released under the
Creative Commons Attribution 4.0 International License ([data/LICENSE](data/LICENSE)).
