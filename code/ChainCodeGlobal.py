from dataclasses import dataclass
from GlobalRingCode import GlobalRingCode


@dataclass
class ChainCodeGlobal(GlobalRingCode):
    """Cyclic code with n = 4d - 7 data qubits on a globally controlled ring, d = 3 or 5.

    See GlobalRingCode for the noise model, measurement sites (k) and detectors.
    """
    def __post_init__(self):
        self.name = 'ChainCode'
        assert self.distance in (3, 5), 'ChainCodeGlobal supports d = 3, 5 (use LinearChainCodeGlobal for 7, 9)'
        self._setup_ring(self.distance ** 2 + 1)

    def syndrome_extraction(self):
        sq, tq = self.single_qubit_gate, self.two_qubit_gate
        sq(['H'], self.ancilla)
        tq('iSWAP')
        sq(['S_DAG', 'H'], self.ancilla)
        for _ in range((self.distance - 1) // 2 - 1):
            tq('SWAP')
        tq('iSWAP')
        tq('iSWAP')
        for _ in range((self.distance - 1) // 2 - 1):
            tq('SWAP')
        sq(['Z', 'H', 'S_DAG'], self.ancilla)
        tq('iSWAP')
        sq(['H'], self.ancilla)
