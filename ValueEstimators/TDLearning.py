import numpy as np
from Agents.Agent import DescentAlgorithm
from Tools import addNewElementForAllActions

class TDLearning(DescentAlgorithm):
    """
    Classic TDLearning algorithm of Sutton [1988] with eligibility traces
    """

    lambda_ = 0        #: lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace_s = None  #: eligibility trace using state only (no copy-paste), necessary for dabney decay mode

    def __init__(self, representation, lambda_ = 0, **kwargs):
        self.representation = representation
        self.eligibility_trace_s= np.zeros(representation.features_num)
        self.lambda_ = lambda_
        super(TDLearning,self).__init__(**kwargs)

    def learn(self, s, a, r, ns, terminal):
        self.representation.pre_discover(s, False, a, ns, terminal)
        gamma = self.representation.domain.gamma
        theta = self.representation.theta
        phi = self.representation.phi(s, False)
        phi_prime = self.representation.phi(ns, terminal)
        nnz = np.count_nonzero(phi)    # Number of non-zero elements

        #Set eligibility traces:
        if self.lambda_:
            # make sure that
            expanded = (- len(self.eligibility_trace_s) + len(phi))
            if expanded > 0:
                # Correct the size of eligibility traces (pad with zeros for new features)
                self.eligibility_trace_s = addNewElementForAllActions(self.eligibility_trace_s, 1, np.zeros((1, expanded)))

            self.eligibility_trace_s  *= gamma*self.lambda_
            self.eligibility_trace_s += phi

            #Set max to 1
            self.eligibility_trace_s[self.eligibility_trace_s>1] = 1
        else:
            self.eligibility_trace_s  = phi

        td_error = r + np.dot(gamma*phi_prime - phi, theta)
        if nnz > 0:
            self.updateAlpha(phi, phi_prime, self.eligibility_trace_s, gamma, nnz, terminal)
            theta_old = theta.copy()
            theta += self.alpha * td_error * self.eligibility_trace_s
            if not np.all(np.isfinite(theta)):
                theta = theta_old
                print "WARNING: TD-Learning diverged, theta reached infinity!"
        #Discover features if the representation has the discover method
        expanded = self.representation.post_discover(s, False, a, td_error, phi)

    def predict(self, s, terminal=False):
        return self.representation.V(s, terminal=terminal, p_actions=None)
