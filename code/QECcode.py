from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import stim


@dataclass
class QECcode(ABC):
    distance: int
    rounds: int
    basis: str = 'Z'
    rounds_before: int = 1
    rounds_after: int = 1
    before_round_data_depolarization: float = 0.
    after_clifford_depolarization_Q1: float = 0.
    after_clifford_depolarization_Q2: float = 0.
    before_measure_flip_probability: float = 0.
    after_reset_flip_probability: float = 0.
    idle_depolarization: float = 0.
    name: str = 'code name'
    k: int = None  # number of measurement sites; None means measure all ancilla at once
    native_2q_gate_set: list = field(default_factory=lambda: ['iSWAP'])  # e.g. ['iSWAP'], ['iSWAP','CZ'], ['iSWAP','SWAP']

    def get_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit += self.initialize_in_basis()
        circuit += self.correction_cycle(rounds=self.rounds_before, noise=False, detector=False)
        for _ in range(rounds):
            circuit += self.correction_cycle(rounds=1, noise=True, detector=True)
        circuit += self.correction_cycle(rounds=self.rounds_after, noise=False, detector=True)
        circuit += self.make_observable()
        return circuit

    def make_observable(self):
        circuit = stim.Circuit()
        circuit.append_operation('M' + self.basis[0], self.data)
        circuit.append_operation('OBSERVABLE_INCLUDE', [stim.target_rec(-1 - i)
                                                        for i in range(len(self.data))])
        return circuit

    def initialize_in_basis(self):
        circuit = stim.Circuit()
        circuit.append_operation('MR' + self.basis[0], self.data)
        return circuit

    @abstractmethod
    def correction_cycle(self, rounds: int, noise: bool, detector: bool) -> stim.Circuit:
        pass

    def shot(self, detector=True, seed=None):
        """
        the stim.Circuit constructs each time
        """
        circuit = self.get_circuit()

        rounds = self.rounds + self.rounds_before + self.rounds_after
        if detector:
            sample = circuit.compile_detector_sampler(seed=seed).sample(1, append_observables=True)[0] * 1
            rounds -= 1
        else:
            sample = circuit.compile_sampler(seed=seed).sample(1)[0] * 1
        res = ''.join('_1'[e] for e in sample)

        print('detectors')
        for i in range(rounds):
            print(res[i * (len(self.ancilla)):(i + 1) * (len(self.ancilla))])
        if detector:
            print('logical')
            print(res[-1])
        else:
            print('data')
            print(res[rounds * (len(self.ancilla)):])
            print('logical')
            print((res[rounds * (len(self.ancilla)):].count('1')) % 2)
            
