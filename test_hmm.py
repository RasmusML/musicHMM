import pytest

import numpy as np
import hmm

def test_viterbi():
    l_trans = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])

    l_emiss = np.array([
        [0.5, 0.5],
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    l_h_init = np.ones(l_trans.shape[0]) / l_trans.shape[0]

    eps = 1e-12

    l_trans = np.log(l_trans + eps)
    l_emiss = np.log(l_emiss + eps)
    l_h_init = np.log(l_h_init + eps)

    seq = np.array([0, 1, 0, 1, 0])

    result_seq = hmm.viterbi(seq, l_trans, l_emiss, l_h_init)

    assert np.all(result_seq == np.array([2, 0, 0, 1, 2]))