from dataclasses import dataclass
from GlobalRingCode import GlobalRingCode


@dataclass
class LinearChainCodeGlobal(GlobalRingCode):
    """Cyclic code with n = 4d - 7 data qubits on a globally controlled ring, d = 7 or 9.

    See GlobalRingCode for the noise model, measurement sites (k) and detectors.
    """
    def __post_init__(self):
        self.name = 'LinearChainCode'
        assert self.distance in (7, 9), 'LinearChainCodeGlobal supports d = 7, 9 (use ChainCodeGlobal for 3, 5)'
        self._setup_ring(8 * self.distance - 14)

    def syndrome_extraction(self):
        sq, tq = self.single_qubit_gate, self.two_qubit_gate
        sq(['H'], self.ancilla)
        tq('iSWAP')
        sq(['S_DAG', 'H'], self.ancilla)
        if self.distance == 7:
            for g in ['SWAP', 'SWAP', 'iSWAP', 'iSWAP', 'iSWAP', 'iSWAP', 'SWAP', 'SWAP']:
                tq(g)
            sq(['H', 'S_DAG'], self.ancilla)
        if self.distance == 9:
            for g in ['iSWAP', 'iSWAP', 'SWAP', 'SWAP', 'iSWAP', 'SWAP', 'SWAP', 'iSWAP',
                      'SWAP', 'SWAP', 'iSWAP', 'iSWAP']:
                tq(g)
            sq(['Z', 'H', 'S_DAG'], self.ancilla)
        tq('iSWAP')
        sq(['H'], self.ancilla)
