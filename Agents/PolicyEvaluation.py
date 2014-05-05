"""BROKEN"""
from .LSPI import LSPI
import Tools
import numy as np
import os.path
__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class PolicyEvaluation(LSPI):

    """presumably an LSTD agent"""
    LOAD_POLICY_FILE = False     # If Q,S,A are read from the file

    def __init__(
            self, representation, policy, domain, logger, sample_window=100,
            accuracy_test_samples=10000, MC_samples=100, target_path='.', re_iterations=100):
        self.compare_with_me = '%s/%s-FixedPolicy.npy' % (target_path,
                                                          Tools.className(domain))
        # Number of iterations over LSPI and iFDD
        self.re_iterations = re_iterations
        super(
            PolicyEvaluation,
            self).__init__(representation,
                           policy,
                           domain,
                           logger,
                           max_window=sample_window + 1,
                           steps_between_LSPI=sample_window,
                           re_iterations=re_iterations)
        # Load the fixedPolicy Estimation if it does not exist create it
        if self.LOAD_POLICY_FILE:
            if not os.path.exists(self.compare_with_me):
                self.logger.log('Generating Fixed Policy Evaluation')
                self.logger.log(
                    'Samples for Accuracy Test = %d' %
                    accuracy_test_samples)
                self.logger.log(
                    'Samples for Monte-Carlo estimation of each Q(s,a) = %d' %
                    MC_samples)
                DATA = self.evaluate(
                    accuracy_test_samples,
                    MC_samples,
                    self.compare_with_me)
            else:
                _, _, shortPolicyFile = self.compare_with_me.rpartition('/')
                DATA = np.load(self.compare_with_me)
                self.logger.log('PE File:\t\t\t%s' % shortPolicyFile)
            self.S = DATA[:, np.arange(self.domain.state_space_dims)]
            self.A = DATA[:, self.domain.state_space_dims].astype(np.uint16)
            self.Q_MC = DATA[:, self.domain.state_space_dims + 1]

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self.process(s, a, r, ns, na, terminal)
        if self.samples_count == self.max_window - 1:
            STATS = []
            start_time = Tools.clock()
            # Representation expansion iteration. Only used if the
            # representation can be expanded
            re_iteration = 0
            added_feature = True
            while added_feature and re_iteration < self.re_iterations:
                # Evaluate the policy and the corresponding PE-Error (Policy
                # Evaluation) and TD-Error
                PE_error, td_errors = self.evaluatePolicy()
                # Save stats
                STATS.append([re_iteration,                     # iteration number
                              self.representation.features_num,
                              # Number of features
                              PE_error,
                             # Policy Evaluation Error
                              # Time since start
                              Tools.deltaT(start_time)
                              ])

                if not Tools.hasFunction(self.representation, 'batchDiscover'):
                    break
                re_iteration += 1
                self.logger.log(
                    'Representation Expansion iteration #%d\n-----------------' %
                    re_iteration)
                added_feature = self.representation.batchDiscover(
                    td_errors,
                    self.all_phi_s,
                    self.data_s)
            self.STATS = np.array(STATS).T  # Experiment will save this later
            self.samples_count = 0
        if terminal:
            self.episodeTerminated()

    def evaluatePolicy(self):
            # Calculate the Q for all samples using the new theta from LSTD
            # 1. newTheta = LSTD
            # 2. build phi_s_a for all samples
            # 3. Q=phi*theta
            # 4. Calculate ||Q-Q_MC||
            # returns all_phi_s (for samples used for LSTD) and td_erros (on
            # samples used for LSTD)

            self.LSTD()
            td_errors = self.calculateTDErrors()
            PE_error = np.linalg.norm(td_errors)

            # Start Calculating the Policy Evaluation Error
#            PE_error_time_start = clock()
#            p                   = self.S.shape[0]
#            n                   = self.representation.features_num
#            test_phi_s          = empty((p,n),dtype=self.representation.featureType())
#            for i in arange(p):
#                test_phi_s[i,:]  = self.representation.phi(self.S[i])
#
#            all_test_phi_s_a    = self.representation.batchPhi_s_a(test_phi_s, self.A,use_sparse=self.use_sparse)
#            Q                   = all_test_phi_s_a * self.representation.theta.reshape(-1,1) if sp.issparse(all_test_phi_s_a) else dot(all_test_phi_s_a,self.representation.theta)
#            PE_error            = linalg.norm(Q.ravel()-self.Q_MC)
#            self.logger.log("||Delta V|| = %f" % PE_error)
            self.logger.log("||TD-Errors|| = %f " % np.linalg.norm(td_errors))
            return PE_error, td_errors
