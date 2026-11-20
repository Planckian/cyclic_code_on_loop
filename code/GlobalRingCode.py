"""
Shared machinery for cyclic codes on a globally controlled ring.

ChainCodeGlobal (d = 3, 5) and LinearChainCodeGlobal (d = 7, 9) only differ in
the syndrome-extraction (SE) gate sequence.  Everything else -- gate
decompositions, noise, serial readout through k measurement sites, stabilizer
bookkeeping and detectors -- lives here so that the two families cannot drift
apart.

Ring layout
-----------
n = 2 * n_syndromes qubits on a ring.  At the start of every cycle (and at every
readout round) data qubits sit on the even sites (self.data) and syndrome
qubits on the odd sites (self.ancilla); this holds because every SE circuit and
every readout transport uses an even number of global two-qubit layers.

Noise model (phenomenological part)
-----------------------------------
p   = before_round_data_depolarization
q   = before_measure_flip_probability
r   = readout_noise_ratio
* every cycle: DEPOLARIZE1(p) on all data qubits (identical to Ref. [33] for M=1)
* every syndrome measurement: X_ERROR(q) just before MR
* after each readout round except the last (i.e. M-1 times per cycle):
  DEPOLARIZE1(r*p) on all data qubits and on the syndrome qubits selected by
  readout_noise_ancillas:
      'all'     -> every syndrome qubit (waiting and already reset)
      'waiting' -> only syndrome qubits not yet measured in this cycle
      'none'    -> no syndrome qubits (data-only readout noise)
  r = 1 with readout_noise_ancillas='none' reproduces the original model.
The circuit-level parameters (after_clifford_depolarization_Q1/Q2,
after_reset_flip_probability) work exactly as before.

Measurement sites
-----------------
k = number of sites with local readout (ell in the paper).  The number of
readout rounds per cycle is M = ceil(n_syndromes / k).  k=None means local
readout everywhere (k = n_syndromes, M = 1).  Use
`cls.k_for_rounds(distance, M)` to get the smallest k that gives M rounds.

Detectors
---------
Each detector compares the same stabilizer in two consecutive cycles.  Its
coordinates are (stabilizer_index, cycle_index), so decoders can recover the
(cycle, stabilizer) layout with circuit.get_detector_coordinates().
"""
from dataclasses import dataclass
import math
import stim
from QECcode import QECcode


@dataclass
class GlobalRingCode(QECcode):
    readout_noise_ratio: float = 1.0
    readout_noise_ancillas: str = 'all'   # 'all' | 'waiting' | 'none'

    # ------------------------------------------------------------------ set-up
    def _setup_ring(self, n: int):
        self.n = n
        self.ancilla = list(range(self.n))[1::2]
        self.data = list(range(self.n))[::2]
        self.pairs0 = list(range(self.n))
        self.pairs1 = list(range(1, self.n)) + [0]
        self.n_syndromes = len(self.ancilla)
        if self.k is None:
            self.k = self.n_syndromes
        if not (1 <= int(self.k) <= self.n_syndromes):
            raise ValueError(f'k must be between 1 and n_syndromes={self.n_syndromes}, got {self.k}')
        self.k = int(self.k)
        if self.readout_noise_ancillas not in ('all', 'waiting', 'none'):
            raise ValueError("readout_noise_ancillas must be 'all', 'waiting' or 'none'")
        if self.readout_noise_ratio < 0:
            raise ValueError('readout_noise_ratio must be >= 0')
        self.meas_sites = self.select_measurement_sites(self.k)
        self._reset_state()

    def _reset_state(self):
        self.address_map = list(range(self.n))   # address_map[site] = qubit label currently at that site
        self.measurement_map = []                # per cycle: syndrome-qubit labels in measurement order
        self._prev_order = None                  # measurement order of the previous cycle
        self._stab_of_label = None               # syndrome-qubit label -> stabilizer index (previous cycle)
        self._cycle_idx = 0

    @property
    def M(self) -> int:
        """Number of readout rounds per SE cycle."""
        return math.ceil(self.n_syndromes / self.k)

    @staticmethod
    def n_syndromes_for(distance: int) -> int:
        return (8 * distance - 14) // 2

    @classmethod
    def k_for_rounds(cls, distance: int, M: int) -> int:
        """Smallest number of measurement sites that reads all syndromes in M rounds."""
        ns = cls.n_syndromes_for(distance)
        k = math.ceil(ns / M)
        if math.ceil(ns / k) != M:
            raise ValueError(f'M={M} is not achievable for d={distance} (n_syndromes={ns})')
        return k

    def select_measurement_sites(self, k):
        """k odd sites spread around the ring (same rule as the original code).

        Every syndrome qubit moves one syndrome slot per readout round, so a site
        reads the M slots that pass it; the assertion below guarantees that the
        gaps between sites are at most M slots, i.e. every syndrome is read.
        """
        spacing = self.n / k
        sites = [1]
        for i in range(1, k):
            rounded = round((1 + i * spacing) % self.n)
            candidate = rounded if rounded % 2 == 1 else rounded + 1
            sites.append(candidate % self.n)
        slots = sorted({(s - 1) // 2 for s in sites})
        assert len(slots) == k, f'duplicate measurement sites for k={k}: {sites}'
        gaps = [((slots[(i + 1) % k] - slots[i]) % self.n_syndromes) or self.n_syndromes for i in range(k)]
        assert max(gaps) <= math.ceil(self.n_syndromes / k), f'sites {sites} cannot read all syndromes'
        return sites

    # ----------------------------------------------------------------- circuit
    def get_circuit(self) -> stim.Circuit:
        self._reset_state()
        circuit = stim.Circuit()
        circuit += self.initialize_in_basis()
        for q in range(self.n):
            circuit.append("QUBIT_COORDS", q, [q])
        for _ in range(self.rounds_before):
            circuit += self._cycle(noise=False, detector=False)
        for _ in range(self.rounds):
            circuit += self._cycle(noise=True, detector=True)
        for _ in range(self.rounds_after):
            circuit += self._cycle(noise=False, detector=True)
        circuit += self.make_observable()
        return circuit

    def correction_cycle(self, rounds: int, noise: bool, detector: bool) -> stim.Circuit:
        """Kept for compatibility. Cycles are generated one at a time so that the
        Python bookkeeping (qubit positions, stabilizer labels) stays in sync."""
        circuit = stim.Circuit()
        for _ in range(rounds):
            circuit += self._cycle(noise=noise, detector=detector)
        return circuit

    def syndrome_extraction(self):
        """Append the SE gate sequence using self.single_qubit_gate / self.two_qubit_gate."""
        raise NotImplementedError

    # --------------------------------------------------------------- gates
    def _update_address_map(self, pairs):
        for q in range(0, self.n, 2):
            a, b = pairs[q], pairs[q + 1]
            self.address_map[a], self.address_map[b] = self.address_map[b], self.address_map[a]

    def _q2_noise(self, pairs):
        p2 = self.after_clifford_depolarization_Q2
        if self._noise and p2 > 0:
            self._circ.append_operation('DEPOLARIZE2', pairs, p2)
        elif self._noise and p2 < 0:   # negative value = single-qubit depolarization of strength |p2|
            self._circ.append_operation('DEPOLARIZE1', pairs, -p2)

    def single_qubit_gate(self, gates, qubits):
        # gates in one call can be combined into one pulse, so noise is applied once
        for gate in gates:
            self._circ.append_operation(gate, qubits)
        if self._noise and self.after_clifford_depolarization_Q1 > 0:
            self._circ.append_operation('DEPOLARIZE1', qubits, self.after_clifford_depolarization_Q1)

    def two_qubit_gate(self, gate):
        if self.two_qubit_gate_cnt % 2 == 0:
            pairs, set1, set2 = self.pairs0, self.ancilla, self.data
        else:
            pairs, set1, set2 = self.pairs1, self.data, self.ancilla
        c = self._circ
        if gate == 'SWAP':
            if 'SWAP' in self.native_2q_gate_set:          # native SWAP
                c.append_operation('SWAP', pairs)
                self._q2_noise(pairs)
            elif 'CZ' in self.native_2q_gate_set:          # SWAP from iSWAP + CZ
                self.single_qubit_gate(['H', 'S_DAG'], set1)
                self.single_qubit_gate(['S_DAG'], set2)
                c.append_operation('iSWAP', pairs)
                self._q2_noise(pairs)
                c.append_operation('CZ', pairs)
                self._q2_noise(pairs)
                self.single_qubit_gate(['H'], set2)
            else:                                          # SWAP from 3 iSWAPs + 3 sqrt(X) layers
                c.append_operation('iSWAP', pairs)
                self._q2_noise(pairs)
                self.single_qubit_gate(['SQRT_X'], set1)
                c.append_operation('iSWAP', pairs)
                self._q2_noise(pairs)
                self.single_qubit_gate(['SQRT_X'], set2)
                c.append_operation('iSWAP', pairs)
                self._q2_noise(pairs)
                self.single_qubit_gate(['SQRT_X'], set1)
        else:                                              # iSWAP (or other native gate)
            c.append_operation(gate, pairs)
            self._q2_noise(pairs)
        self.two_qubit_gate_cnt += 1
        if gate in ('iSWAP', 'SWAP'):
            self._update_address_map(pairs)

    # --------------------------------------------------------------- one cycle
    def _cycle(self, noise: bool, detector: bool) -> stim.Circuit:
        self._circ = circuit = stim.Circuit()
        self._noise = noise
        self.two_qubit_gate_cnt = 0
        p = self.before_round_data_depolarization

        # phenomenological data noise, once per cycle
        if noise and p > 0:
            circuit.append_operation('DEPOLARIZE1', self.data, p)

        self.syndrome_extraction()

        # serial readout: measure at the k sites, then shift syndromes by one slot
        measured = []
        n_m = self.M
        for m in range(n_m):
            to_measure = []
            for site in self.meas_sites:
                label = self.address_map[site]
                if label not in measured:
                    to_measure.append(site)
                    measured.append(label)
            if noise and self.before_measure_flip_probability > 0:
                circuit.append_operation('X_ERROR', to_measure, self.before_measure_flip_probability)
            circuit.append_operation('MR', to_measure)
            if noise and self.after_reset_flip_probability > 0:
                circuit.append_operation('X_ERROR', to_measure, self.after_reset_flip_probability)

            if m < n_m - 1:
                self.two_qubit_gate('SWAP')
                self.two_qubit_gate('SWAP')
                p_r = self.readout_noise_ratio * p
                if noise and p_r > 0:
                    targets = list(self.data)
                    if self.readout_noise_ancillas == 'all':
                        targets += self.ancilla
                    elif self.readout_noise_ancillas == 'waiting':
                        targets += [s for s in self.ancilla if self.address_map[s] not in measured]
                    circuit.append_operation('DEPOLARIZE1', targets, p_r)

        assert len(measured) == self.n_syndromes, 'not every syndrome was measured this cycle'
        self.measurement_map.append(list(measured))

        # stabilizer bookkeeping: the syndrome qubit with label L in the previous
        # cycle and the one with label L + 2*shift in this cycle measure the same
        # stabilizer, where shift = (two-qubit layers per cycle) mod n_syndromes.
        two_ns = 2 * self.n_syndromes
        if self._stab_of_label is None:
            stab_of_label = {a: (a - 1) // 2 for a in self.ancilla}
            pairs = []
        else:
            shift = self.two_qubit_gate_cnt % self.n_syndromes
            stab_of_label, pairs = {}, []
            for old, s in self._stab_of_label.items():
                new = (old + 2 * shift) % two_ns
                stab_of_label[new] = s
                pairs.append((s, old, new))

        if detector:
            if self._prev_order is None:
                raise ValueError('detectors need a previous cycle: use rounds_before >= 1')
            ns = self.n_syndromes
            prev_pos = {lab: j for j, lab in enumerate(self._prev_order)}
            cur_pos = {lab: j for j, lab in enumerate(measured)}
            for s, old, new in sorted(pairs):
                circuit.append('DETECTOR',
                               [stim.target_rec(prev_pos[old] - 2 * ns), stim.target_rec(cur_pos[new] - ns)],
                               [s, self._cycle_idx])

        self._stab_of_label = stab_of_label
        self._prev_order = list(measured)
        self._cycle_idx += 1
        return circuit
