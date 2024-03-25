import numpy as np
from scipy.special import logsumexp

class HMM:
    def __init__(self, n_hidden, n_obs):
        self.n_hidden = n_hidden
        self.n_obs = n_obs

    def fit(self, seq, n_iter=100, l_h_init=None, l_trans_init=None, l_emiss_init=None):
        self._l_trans, self._l_emiss, self._l_h_init, self._history = baum_welch(seq, self.n_hidden, self.n_obs, n_iter, l_h_init=l_h_init, l_trans=l_trans_init, l_emiss=l_emiss_init)

    def predict(self, seq):
        seq, _= viterbi(seq, self._l_trans, self._l_emiss, self._l_h_init)
        return seq
    
    def sample(self, length):
        seq = np.zeros(length, dtype=np.int32)
        hidden = np.zeros(length, dtype=np.int32)

        emissions = np.exp(self._l_emiss)
        transitions = np.exp(self._l_trans)

        hidden[0] = np.random.choice(self.n_hidden, p=np.exp(self._l_h_init))
        seq[0] = np.random.choice(self.n_obs, p=emissions[hidden[0]])

        for t in range(1, length):
            hidden[t] = np.random.choice(self.n_hidden, p=transitions[hidden[t-1]])
            seq[t] = np.random.choice(self.n_obs, p=emissions[hidden[t]])

        return seq, hidden
    
    def get_transition_matrix(self):
        return np.exp(self._l_trans)
    
    def get_emission_matrix(self):
        return np.exp(self._l_emiss)
    
    def get_initial_probabilities(self):
        return np.exp(self._l_h_init)


def forward(seq, l_trans, l_emiss, l_h_init):
    # p(zn, x1:n)

    n_t = seq.shape[0]
    n_h = l_trans.shape[0]

    l_alpha = np.zeros((n_h, n_t))
    l_alpha[:,0] = l_h_init + l_emiss[:,seq[0]]
    
    for t in range(1, n_t):
        l_alpha[:,t] = l_emiss[:,seq[t]] + logsumexp(l_trans + l_alpha[:,t-1].reshape(-1,1), axis=0)
        
    return l_alpha


def backward(seq, l_trans, l_emiss):
    # p(x_{t+1:n} | zn)

    n_t = seq.shape[0]
    n_h = l_trans.shape[0]

    l_beta = np.zeros((n_h, n_t))
    l_beta[:,-1] = 0

    for t in reversed(range(n_t-1)):
        l_beta[:,t] = logsumexp(l_beta[:,t+1] + l_emiss[:,seq[t+1]] + l_trans, axis=1)
        
    return l_beta


def baum_welch(seq, n_h, n_obs, n_iter, l_h_init=None, l_trans=None, l_emiss=None, convergence_eps=1e-5):
    """EM algorithm for HMMs.
    Numerical instabilities are handled by working in log-space.

    References:
    - https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Algorithm
    - https://gregorygundersen.com/blog/2019/11/10/em/
    - https://gregorygundersen.com/blog/2020/11/28/hmms/
    """
    # init
    n_t = seq.shape[0]

    if l_h_init is None:
        l_h_init = np.ones(n_h) / n_h
        l_h_init = np.log(l_h_init)

    if l_trans is None:
        l_trans = np.random.normal(size=(n_h, n_h))
        l_trans -= logsumexp(l_trans, axis=0)

    if l_emiss is None:
        l_emiss = np.random.normal(size=(n_h, n_obs))
        l_emiss -= logsumexp(l_emiss, axis=0)

    lls = []

    ### EM
    for i in range(n_iter):
        ## E-step
        l_alpha = forward(seq, l_trans, l_emiss, l_h_init)  # p(zn, x1:n)
        l_beta = backward(seq, l_trans, l_emiss)            # p(x_{t+1:n} | zn)
        
        # compute log-likelihood
        ll = logsumexp(l_alpha[:,-1])
        lls += [ll]

        l_gamma = l_alpha + l_beta            # p(zn, x1:n)
        l_gamma -= logsumexp(l_gamma, axis=0) # p(zn | x1:n)

        # p(zn, zn+1 | x1:n)
        l_xi = np.zeros((n_h, n_h, n_t-1))
        for t in range(n_t-1):
            w = l_alpha[:,t].reshape(-1,1) + l_trans + l_emiss[:,seq[t+1]] + l_beta[:,t+1]
            l_xi[:,:,t] = w - logsumexp(w)
                
        ## M-step
        l_h_init = l_gamma[:,0]
        assert np.isclose(np.sum(np.exp(l_h_init)), 1), f"{np.sum(np.exp(l_h_init))}"

        for i in range(n_obs):
            l_emiss[:,i] = logsumexp(l_gamma[:,seq==i], axis=1) - logsumexp(l_gamma, axis=1) if np.sum(seq==i) > 0 else -100
        
        assert np.all(np.isclose(np.sum(np.exp(l_emiss), axis=1), 1)), f"{np.sum(np.exp(l_emiss), axis=1)}"
        
        for i in range(n_h):
            l_trans[i] = logsumexp(l_xi[i], axis=1) - logsumexp(l_xi[i])
        
        assert np.all(np.isclose(np.sum(np.exp(l_trans), axis=1), 1)), f"{np.sum(np.exp(l_trans), axis=1)}"

        # check for convergence
        #if len(lls) > 1 and np.abs(lls[-1] - lls[-2]) < convergence_eps:
        #    break

    return l_trans, l_emiss, l_h_init, lls


def viterbi(seq, l_trans, l_emiss, l_h_init):
    n_t = seq.shape[0]
    n_h = l_trans.shape[0]

    prob = np.zeros((n_h, n_t))
    prev = -np.ones((n_h, n_t), dtype=np.int32)

    # compute probs
    prob[:,0] = l_h_init + l_emiss[:,seq[0]]
    for t in range(1, n_t):
        pn = prob[:,t-1].reshape(-1,1) + l_trans + l_emiss[:,seq[t]]
        prob[:,t] = np.max(pn, axis=0)
        prev[:,t] = np.argmax(pn, axis=0)

    # extract most likely hidden states
    seq_h = np.zeros(n_t, dtype=np.int32)
    seq_h[-1] = np.argmax(prob[:,-1])
    for t in range(1, n_t):
        seq_h[-t-1] = prev[seq_h[-t],-t]

    return seq_h, prob