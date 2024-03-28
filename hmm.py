import numpy as np
from scipy.special import logsumexp

class HMM:
    def __init__(self, n_hidden, n_obs):
        self.n_hidden = n_hidden
        self.n_obs = n_obs

    def fit(self, seqs, lengths, n_iter=100, l_h_init=None, l_trans_init=None, l_emiss_init=None, convergence_eps=-1):
        self._l_trans, self._l_emiss, self._l_h_init, self._history = baum_welch(seqs, lengths, self.n_hidden, self.n_obs, n_iter, l_h_init=l_h_init, l_trans=l_trans_init, l_emiss=l_emiss_init, convergence_eps=convergence_eps)

    def predict(self, hidden_seq):
        hidden_seq, _= viterbi(hidden_seq, self._l_trans, self._l_emiss, self._l_h_init)
        return hidden_seq
    
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


def baum_welch(seqs, lengths, n_hidden, n_obs, n_iter, l_h_init=None, l_trans=None, l_emiss=None, convergence_eps=-1):
    """EM algorithm for HMMs.
    Numerical instabilities are handled by working in log-space.

    References:
    - https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Algorithm
    - https://gregorygundersen.com/blog/2019/11/10/em/
    - https://gregorygundersen.com/blog/2020/11/28/hmms/
    """
    ## init
    
    if l_h_init is None:
        l_h_init = np.log(np.ones(n_hidden) / n_hidden)

    if l_trans is None:
        probs = np.random.dirichlet(np.ones(n_hidden), size=n_hidden)
        l_trans = np.log(probs)

    if l_emiss is None:
        probs = np.random.dirichlet(np.ones(n_obs), size=n_hidden)
        l_emiss = np.log(probs)

    lls = []

    n_seqs = len(lengths)
    
    seqs_start_inclusive = np.zeros(n_seqs, dtype=np.int32)
    seqs_start_inclusive[1:] = np.cumsum(lengths)[:-1]

    seqs_end_exclusive = np.cumsum(lengths)

    total_length = np.sum(lengths)

    seqs_mask = np.zeros((n_obs, total_length), dtype=bool)
    seqs_mask[seqs, np.arange(total_length)] = True

    ### EM
    for i in range(n_iter):
        ## E-step
        ll = 0

        l_gamma_all = np.zeros((n_hidden, total_length))
        l_xi_all = np.zeros((n_hidden, n_hidden, total_length-1))

        for s, e in zip(seqs_start_inclusive, seqs_end_exclusive):
            seq = seqs[s:e]
            seq_len = seq.shape[0]

            l_alpha = forward(seq, l_trans, l_emiss, l_h_init)  # p(zn, x1:n)
            l_beta = backward(seq, l_trans, l_emiss)            # p(x_{t+1:n} | zn)
            
            l_gamma = l_alpha + l_beta            # p(zn, x1:n)
            l_gamma -= logsumexp(l_gamma, axis=0) # p(zn | x1:n)

            # p(zn, zn+1 | x1:n)
            l_xi = np.zeros((n_hidden, n_hidden, seq_len-1))
            for t in range(seq_len-1):
                w = l_alpha[:,t].reshape(-1,1) + l_trans + l_emiss[:,seq[t+1]] + l_beta[:,t+1]
                l_xi[:,:,t] = w - logsumexp(w)

            l_gamma_all[:,s:e] = l_gamma
            l_xi_all[:,:,s:e-1] = l_xi

            # compute log-likelihood
            ll += logsumexp(l_alpha[:,-1])

        lls.append(ll)

        ## M-step
        l_h_init = logsumexp(l_gamma_all[:,seqs_start_inclusive], axis=1) - np.log(n_seqs)
        assert np.isclose(np.sum(np.exp(l_h_init)), 1), f"{np.sum(np.exp(l_h_init))}"

        for i in range(n_obs):
            m = seqs_mask[i]
            l_emiss[:,i] = logsumexp(l_gamma_all[:,m], axis=1) - logsumexp(l_gamma_all, axis=1) if np.sum(m) > 0 else -100
        
        assert np.all(np.isclose(np.sum(np.exp(l_emiss), axis=1), 1)), f"{np.sum(np.exp(l_emiss), axis=1)}"
        
        for i in range(n_hidden):
            l_trans[i] = logsumexp(l_xi_all[i], axis=1) - logsumexp(l_xi_all[i])
        
        assert np.all(np.isclose(np.sum(np.exp(l_trans), axis=1), 1)), f"{np.sum(np.exp(l_trans), axis=1)}"

        # check for convergence
        if len(lls) > 1 and np.abs(lls[-1] - lls[-2]) < convergence_eps:
            break

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
