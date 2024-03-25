import pytest

import numpy as np
import hmm
from scipy.special import logsumexp


def test_viterbi():
    p0 = np.array([0.6, 0.4])
    transition = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    emissions = np.array([
        [0.5, 0.1], 
        [0.4, 0.3], 
        [0.1, 0.6]
    ])

    seq = np.array([0, 0, 1, 2, 2])

    l_trans = np.log(transition)
    l_emiss = np.log(emissions.T)
    l_p0 = np.log(p0)
    seq, l_probs = hmm.viterbi(seq, l_trans, l_emiss, l_p0)

    assert np.all(seq == np.array([0, 0, 0, 1, 1]))    
    
    # using the results of this HMM as expected values:
    # https://github.com/nwams/Hidden_Markov_Model-in-PyTorch/blob/master/Hidden%20Markov%20Models%20in%20PyTorch.ipynb
    probs_actual = np.exp(l_probs)
    probs_expected = np.array([
        [0.3, 0.105, 0.0294, 0.002058, 0.00021168],
        [0.04, 0.009, 0.00945, 0.005292, 0.00190512]
    ])

    assert np.allclose(probs_actual, probs_expected)


def test_forward():
    """
    test the forward algorithm: alpha := p(zn, x1:n)

    HMM with two hidden states and two observations.

    Assume the following graph:
        a - b
        |   |
        0   1
    """
    # [a b]
    l_h_init = np.array([0.4, 0.6])

    # [a b] -> [a b]
    l_trans = np.array([
        [0.3, 0.7],
        [0.4, 0.6]
    ])

    # [a b] -> [0 1]
    l_emiss = np.array([
        [0.3, 0.7],
        [0.1, 0.9]
    ])

    l_h_init = np.log(l_h_init)
    l_trans = np.log(l_trans)
    l_emiss = np.log(l_emiss)

    hidden = np.array([0, 1]) # [z1=a, z2=b]
    seq = np.array([0, 0])    # [x1=0, x2=0]

    alpha_forward = hmm.forward(seq, l_trans, l_emiss, l_h_init)

    # extract alpha for the hidden states: [z1=a, z2=b]
    alpha_actual = np.array([alpha_forward[hidden[0],0], alpha_forward[hidden[1],1]])

    # manual calculations of alpha
    alpha_expected = np.array([
        l_h_init[hidden[0]] + l_emiss[hidden[0],seq[0]],
        l_emiss[hidden[1],seq[1]] + logsumexp(np.array([
            l_trans[0,hidden[1]] + l_h_init[0] + l_emiss[0,seq[0]],
            l_trans[1,hidden[1]] + l_h_init[1] + l_emiss[1,seq[0]]
        ]))
    ])

    assert np.allclose(alpha_expected, alpha_actual)


def test_forward2():
    p0 = np.array([0.5, 0.5])

    emissions = np.array([
        [0.9, 0.2], 
        [0.1, 0.8]
    ])

    transition = np.array([
        [0.7, 0.3],
        [0.3, 0.7]]
    )

    seq = np.array([1, 1, 0, 0, 0, 1])

    l_trans = np.log(transition)
    l_emiss = np.log(emissions.T)
    l_p0 = np.log(p0)
    l_alpha = hmm.forward(seq, l_trans, l_emiss, l_p0)

    alpha_actual = np.exp(l_alpha)
    alpha_actual /= alpha_actual.sum(axis=0)

    # using the results of this HMM as expected values:
    # https://github.com/nwams/Hidden_Markov_Model-in-PyTorch/blob/master/Hidden%20Markov%20Models%20in%20PyTorch.ipynb
    alpha_expected = np.array([
        [0.11111, 0.06163, 0.68386, 0.85819, 0.89029, 0.19256],
        [0.88888, 0.93837, 0.31613, 0.14180, 0.10971, 0.80743]
    ])
    alpha_expected /= alpha_expected.sum(axis=0)

    assert np.allclose(alpha_actual, alpha_expected, atol=1e-5)


def test_backward():
    """
    test the backward algorithm: beta := p(xn+1:T | zn)

    HMM with two hidden states and two observations.

    Assume the following graph:
        a - b
        |   |
        0   1
    """
    # [a b]
    l_h_init = np.array([0.4, 0.6])

    # [a b] -> [a b]
    l_trans = np.array([
        [0.3, 0.7],
        [0.4, 0.6]
    ])

    # [a b] -> [0 1]
    l_emiss = np.array([
        [0.3, 0.7],
        [0.1, 0.9]
    ])

    l_h_init = np.log(l_h_init)
    l_trans = np.log(l_trans)
    l_emiss = np.log(l_emiss)

    hidden = np.array([0, 1]) # [z1=a, z2=b]
    seq = np.array([0, 0])    # [x1=0, x2=0]

    beta_forward = hmm.backward(seq, l_trans, l_emiss)

    # extract alpha for the hidden states: [z1=a, z2=b]
    beta_actual = np.array([beta_forward[hidden[0],0], beta_forward[hidden[1],1]])

    # manual calculations of alpha
    beta_expected = np.array([
        logsumexp(np.array([
            l_emiss[0,seq[1]] + l_trans[hidden[0],0],
            l_emiss[1,seq[1]] + l_trans[hidden[0],1]
        ])),
        0,
    ])

    assert np.allclose(beta_expected, beta_actual)


def test_backward2():
    emissions = np.array([
        [0.9, 0.2], 
        [0.1, 0.8]
    ])

    transition = np.array([
        [0.7, 0.3],
        [0.3, 0.7]]
    )

    seq = np.array([1, 1, 0, 0, 0, 1])

    l_trans = np.log(transition)
    l_emiss = np.log(emissions.T)
    l_beta = hmm.backward(seq, l_trans, l_emiss)

    beta_actual = np.exp(l_beta)
    beta_actual /= beta_actual.sum(axis=0)

    # using the results of this HMM as expected values:
    # https://github.com/nwams/Hidden_Markov_Model-in-PyTorch/blob/master/Hidden%20Markov%20Models%20in%20PyTorch.ipynb
    beta_expected = np.array([
        [1.40699, 3.23616, 2.72775, 1.73108, 1.39911, 2.93497],
        [2.32412, 1.69423, 1.50282, 1.24784, 2.66283, 2.93497]
    ])
    beta_expected /= beta_expected.sum(axis=0)

    assert np.allclose(beta_actual, beta_expected, atol=1e-5)


def test_posterior():
    p0 = np.array([0.5, 0.5])

    emissions = np.array([
        [0.9, 0.2], 
        [0.1, 0.8]
    ])

    transition = np.array([
        [0.7, 0.3],
        [0.3, 0.7]]
    )

    seq = np.array([1, 1, 0, 0, 0, 1])

    l_trans = np.log(transition)
    l_emiss = np.log(emissions.T)
    l_p0 = np.log(p0)

    l_alpha = hmm.forward(seq, l_trans, l_emiss, l_p0)
    l_beta = hmm.backward(seq, l_trans, l_emiss)

    l_gamma = l_alpha + l_beta
    l_gamma -= logsumexp(l_gamma, axis=0)

    posterior_actual = np.exp(l_gamma)

    # using the results of this HMM as expected values:
    # https://github.com/nwams/Hidden_Markov_Model-in-PyTorch/blob/master/Hidden%20Markov%20Models%20in%20PyTorch.ipynb
    alpha_expected = np.array([
        [0.11111, 0.06163, 0.68386, 0.85819, 0.89029, 0.19256],
        [0.88888, 0.93837, 0.31613, 0.14180, 0.10971, 0.80743]
    ])
    beta_expected = np.array([
        [1.40699, 3.23616, 2.72775, 1.73108, 1.39911, 2.93497],
        [2.32412, 1.69423, 1.50282, 1.24784, 2.66283, 2.93497]
    ])
    posterior_expected = alpha_expected * beta_expected
    posterior_expected /= posterior_expected.sum(axis=0)

    assert np.allclose(posterior_expected, posterior_actual, atol=1e-5)
