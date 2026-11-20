import numpy as np


class LookupTableDecoder:
    def __init__(self, code, lookuptable=None):
        '''
        Only ChainCode, LinearChainCode and SurfaceCode supported

        Input to decode_batch has shape (shots, cycles, n_syndromes) with the
        stabilizers in cyclic order.  With detector=True it must be detection
        events whose first row is all zeros (collectorGlobal.detectors_for_lut
        builds exactly this); with detector=False it is raw syndrome outcomes.
        '''
        nq = len(code.data)
        ne = (code.distance - 1) // 2
        self.basis = code.basis

        if code.name in ('ChainCode', 'ChainCodeGlobal'):
            syn_table = []
            for syn_id in range((code.distance ** 2 + 1) // 2):
                syn = np.zeros((code.distance ** 2 + 1) // 2)
                s = np.array(
                    [3] + [0] * ((code.distance - 1) // 2 - 1) + [1, 1] + [0] * ((code.distance - 1) // 2 - 1) + [3])
                syn[0:len(s)] = s
                syn_table.append(np.roll(syn, syn_id))
            self.syn_table = np.asarray(syn_table)
        elif code.name in ('SurfaceCode',):
            self.syn_table = np.zeros((2 * len(code.xancillaqubits), code.distance ** 2), dtype=int)
            for i in range(len(code.xancillaqubits)):
                for j in code.xancillaqubits[i]:
                    if j != -1:
                        self.syn_table[i, j] = 1
                for j in code.zancillaqubits[i]:
                    if j != -1:
                        self.syn_table[len(code.xancillaqubits) + i, j] = 3
        elif code.name in ('LinearChainCode', 'LinearChainCodeGlobal'):
            if code.distance == 7:
                syn_table = []
                for syn_id in range(code.n//2):
                    syn = np.zeros(code.n//2)
                    s = np.array([3, 0, 0, 1, 1, 1, 1, 0, 0, 3])
                    # s = np.array([1, 0, 0, 3, 3, 3, 3, 0, 0, 1])
                    syn[0:len(s)] = s
                    syn_table.append(np.roll(syn, syn_id))
                self.syn_table = np.asarray(syn_table)
            if code.distance == 9:
                syn_table = []
                for syn_id in range(code.n//2):
                    syn = np.zeros(code.n//2)
                    s = np.array([3, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 3])
                    # s = np.array([1, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 3, 1])
                    syn[0:len(s)] = s
                    syn_table.append(np.roll(syn, syn_id))
                self.syn_table = np.asarray(syn_table)
        else:
            raise ValueError(f"LookupTableDecoder does not support code '{code.name}'")

        if lookuptable is not None:
            # precomputed table (e.g. a memory-mapped file shared by several worker processes)
            self.lookuptable = lookuptable
            return

        errors = [i for i in generate_errors(nq, ne, True)]
        error_syndromes = [tuple(check_syndromes(self.syn_table, i)) for i in errors]
        logical_errors = [logical_error(i) for i in errors]
        error_syndromes = np.asarray(error_syndromes)
        for i in range(error_syndromes.shape[1]):
            error_syndromes[:, i] *= 2 ** i
        error_indeces = np.sum(error_syndromes, axis=1)
        # int8 is enough (entries are 0-3) and keeps the d=9 table at ~0.5 GB instead of ~4 GB
        self.lookuptable = np.zeros((np.max(error_indeces) + 1), np.int8)
        for index, lerror in zip(error_indeces, logical_errors):
            self.lookuptable[index] = lerror

    def preproc_m_ancilla(self, m_ancilla):
        ancillas = np.zeros(m_ancilla.shape, dtype=int)
        ancillas[:, 1:] = (m_ancilla[:, 1:] - m_ancilla[:, :-1]) % 2
        return ancillas

    def preproc4betterLUT(self, preproc_m_ancilla):
        runs, cycles, n = preproc_m_ancilla.shape
        for i in range(runs):
            for j in range(1, cycles - 1):
                preproc_m_ancilla_sum = np.sum(preproc_m_ancilla[i, j])
                preproc_m_ancilla_sum1 = np.sum(preproc_m_ancilla[i, j + 1])
                if preproc_m_ancilla_sum != 0 and preproc_m_ancilla_sum1 != 0:
                    preproc_m_ancilla[i, j + 1] = (preproc_m_ancilla[i, j] + preproc_m_ancilla[i, j + 1]) % 2
                    preproc_m_ancilla[i, j] *= 0
        return preproc_m_ancilla

    def preproc4betterLUT2(self, preproc_m_ancilla):
        runs, cycles, n = preproc_m_ancilla.shape
        for i in range(runs):
            for j in range(1, cycles - 1):
                preproc_m_ancilla_sum = np.sum(preproc_m_ancilla[i, j])
                preproc_m_ancilla_sum1 = np.sum(preproc_m_ancilla[i, j + 1])
                if j < cycles - 2:
                    preproc_m_ancilla_sum2 = np.sum(preproc_m_ancilla[i, j + 2])
                    if preproc_m_ancilla_sum != 0 and preproc_m_ancilla_sum1 + preproc_m_ancilla_sum2 != 0:
                        preproc_m_ancilla[i, j + 1] = (preproc_m_ancilla[i, j] + preproc_m_ancilla[i, j + 1]) % 2
                        preproc_m_ancilla[i, j] *= 0
                elif preproc_m_ancilla_sum != 0 and preproc_m_ancilla_sum1 != 0:
                    preproc_m_ancilla[i, j + 1] = (preproc_m_ancilla[i, j] + preproc_m_ancilla[i, j + 1]) % 2
                    preproc_m_ancilla[i, j] *= 0
        return preproc_m_ancilla

    def decode_batch(self, m_ancilla, better_preproc=False, better_preproc2=False,
                     detector=True, correct_history=False, batchsize=10000):
        # work on an int64 copy: decode_ multiplies in place by 2**i, which would
        # silently overflow for bool/uint8 input and would modify the caller's array
        m_ancilla = np.array(m_ancilla, dtype=np.int64, copy=True)
        correct_m_data = np.zeros((m_ancilla.shape[0], m_ancilla.shape[1]), dtype=int)
        for i in range(0, m_ancilla.shape[0], batchsize):
            if not detector:
                ancillas = self.preproc_m_ancilla(m_ancilla[i:i + batchsize])
            else:
                ancillas = m_ancilla[i:i + batchsize]
            if better_preproc:
                ancillas = self.preproc4betterLUT(ancillas)
            assert better_preproc == False or better_preproc2 == False
            if better_preproc2:
                ancillas = self.preproc4betterLUT2(ancillas)
            correct_m_data[i:i + batchsize] = self.decode_(ancillas)
        if correct_history:
            return correct_m_data
        else:
            return correct_m_data[:, -1]

    def decode_(self, ancillas):
        for i in range(ancillas.shape[-1]):
            ancillas[:, :, i] *= 2 ** i
        ancillas = np.sum(ancillas, axis=-1)
        assert np.all(ancillas >= 0), 'ancillas dtype should be longer'
        lerrors = np.zeros(ancillas.shape, dtype=int)
        for i in range(ancillas.shape[0]):
            if np.sum(ancillas[i]) > 0:
                x_errors_cnt = 0
                z_errors_cnt = 0
                for j in range(ancillas.shape[1]):
                    if ancillas[i, j] < len(self.lookuptable):
                        if self.lookuptable[ancillas[i, j]] == 1:
                            x_errors_cnt += 1
                        elif self.lookuptable[ancillas[i, j]] == 2:
                            x_errors_cnt += 1
                            z_errors_cnt += 1
                        elif self.lookuptable[ancillas[i, j]] == 3:
                            z_errors_cnt += 1
                    if x_errors_cnt % 2 == 1 and z_errors_cnt % 2 == 0:
                        lerrors[i, j] = 1
                    elif x_errors_cnt % 2 == 1 and z_errors_cnt % 2 == 1:
                        lerrors[i, j] = 2
                    elif x_errors_cnt % 2 == 0 and z_errors_cnt % 2 == 1:
                        lerrors[i, j] = 3
        if self.basis == 'Z':
            error = [1, 2]
        elif self.basis == 'X':
            error = [2, 3]
        elif self.basis == 'Y':
            error = [1, 3]
        correct_m_data = ((lerrors == error[0]).astype(int) + (lerrors == error[1]).astype(int)) % 2
        return correct_m_data


def generate_errors(nq, ne, cum=False):
    if cum and ne > 0:
        for error in generate_errors(nq, ne - 1, True):
            yield error
    for error_type in range(1, 4):
        if ne > 1:
            for rest_errors in generate_errors(nq - 1, ne - 1):
                yield [error_type] + rest_errors
        elif ne == 1:
            yield [error_type] + [0] * (nq - 1)
    if ne < nq:
        for rest_errors in generate_errors(nq - 1, ne):
            yield [0] + rest_errors


def check_syndromes(syndromes, error):
    return np.sum((1 - (syndromes == error)) * (syndromes > 0) * (np.asarray(error) > 0), axis=1) % 2


def logical_error(error):
    le = 0
    for qe in error:
        if qe == 0:
            pass
        elif le == 0:
            le = qe
        elif le == qe:
            le = 0
        else:
            le = [e for e in [1, 2, 3] if e not in [le, qe]][0]
    return le