#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Feb 22th 2013 at MIT #
######################################################
# This agent fits the parameter to the gathered data using a fixed policy with LSTD algorithm.
# The accuracy of the fit is calculated as follows:
# target_path points to a directory containing: <Domain-Name>-FixedPolicy.txt 
# contains a set of samples in form of s,a,Q_MC
# where Q_MC is calculated using monte-carlo sampling
# The accuracy will be L2 norm of Q_MC and Q approximated using the output of the LSTD over all the samples 
# Note if <compare_with_me> file does not exist on the first iteration it will be filled automatically and saved.
# accuracy_test_samples: Number of s,a,Q_MC(s,a)
# MC_samples: Number of samples used to estimate Q_MC(s,a)
from LSPI import *
class PolicyEvaluation(LSPI):
    LOAD_POLICY_FILE = False     # If Q,S,A are read from the file
    def __init__(self,representation,policy,domain,logger, sample_window = 100, accuracy_test_samples = 10000, MC_samples = 100, target_path = '.',re_iterations = 100):
        self.compare_with_me = '%s/%s-FixedPolicy.npy' %(target_path,className(domain))
        self.re_iterations  = re_iterations # Number of iterations over LSPI and iFDD
        super(PolicyEvaluation,self).__init__(representation,policy,domain,logger, max_window = sample_window, steps_between_LSPI = sample_window, re_iterations = re_iterations)
        # Load the fixedPolicy Estimation if it does not exist create it
        if self.LOAD_POLICY_FILE:
            if not os.path.exists(self.compare_with_me):
                self.logger.log('Generating Fixed Policy Evaluation')
                self.logger.log('Samples for Accuracy Test = %d' % accuracy_test_samples)
                self.logger.log('Samples for Monte-Carlo estimation of each Q(s,a) = %d' % MC_samples)
                DATA = self.evaluate(accuracy_test_samples, MC_samples, self.compare_with_me)
            else:
                _,_,shortPolicyFile =  self.compare_with_me.rpartition('/')
                DATA = load(self.compare_with_me)
                self.logger.log('PE File:\t\t\t%s' % shortPolicyFile)
            self.S      = DATA[:,arange(self.domain.state_space_dims)]
            self.A      = DATA[:,self.domain.state_space_dims].astype(uint16)
            self.Q_MC   = DATA[:,self.domain.state_space_dims+1]
    def learn(self,s,a,r,ns,na,terminal):
        self.process(s,a,r,ns,na,terminal)
        if self.samples_count == self.max_window:
            STATS               = [] 
            start_time          = clock()
            re_iteration        = 0 # Representation expansion iteration. Only used if the representation can be expanded 
            added_feature       = True
            while added_feature and re_iteration < self.re_iterations:
                # Evaluate the policy and the corresponding PE-Error (Policy Evaluation) and TD-Error
                PE_error, td_errors  = self.evaluatePolicy()
                # Save stats
                STATS.append([re_iteration,                     # iteration number
                              self.representation.features_num, # Number of features
                              PE_error,                         # Policy Evaluation Error
                              deltaT(start_time)                # Time since start
                              ])
                
                if not hasFunction(self.representation,'batchDiscover'):
                    break
                re_iteration += 1
                self.logger.log('Representation Expansion iteration #%d\n-----------------' % re_iteration)
                added_feature = self.representation.batchDiscover(td_errors, self.all_phi_s, self.data_s)
            self.STATS = array(STATS).T # Experiment will save this later
    def evaluatePolicy(self):
            #Calculate the Q for all samples using the new theta from LSTD
            #1. newTheta = LSTD
            #2. build phi_s_a for all samples
            #3. Q=phi*theta
            #4. Calculate ||Q-Q_MC||
            # returns all_phi_s (for samples used for LSTD) and td_erros (on samples used for LSTD)
            
            self.LSTD()
            td_errors = self.calculateTDErrors()            
            PE_error  = linalg.norm(td_errors)
            
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
            self.logger.log("||TD-Errors|| = %f " % linalg.norm(td_errors))
            return PE_error, td_errors 
