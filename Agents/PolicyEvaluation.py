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
    def __init__(self,representation,policy,domain,logger, sample_window = 100, accuracy_test_samples = 10000, MC_samples = 100, target_path = '.',re_iterations = 100):
        self.compare_with_me = '%s/%s-FixedPolicy.npy' %(target_path,className(domain))
        self.re_iterations  = re_iterations # Number of iterations over LSPI and iFDD
        super(PolicyEvaluation,self).__init__(representation,policy,domain,logger, sample_window = sample_window)
        # Load the fixedPolicy Estimation if it does not exist create it
        if not os.path.exists(self.compare_with_me):
            self.logger.log('Generating Fixed Policy Evaluation')
            self.logger.log('Samples for Accuracy Test = %d' % accuracy_test_samples)
            self.logger.log('Samples for Monte-Carlo estimation of each Q(s,a) = %d' % MC_samples)
            DATA = self.evaulate(accuracy_test_samples, MC_samples, self.compare_with_me)
        else:
            self.logger.log('Loading Fixed Policy Evaluation from %s:' % self.compare_with_me)
            DATA = load(self.compare_with_me)
        self.S      = DATA[:,arange(self.domain.state_space_dims)]
        self.A      = DATA[:,self.domain.state_space_dims].astype(uint16)
        self.Q_MC   = DATA[:,self.domain.state_space_dims+1]
        if logger and hasFunction(self.representation,'batchDiscover'):
            logger.log('Max Representation Expansion Iterations:\t%d' % re_iterations)
    def learn(self,s,a,r,ns,na,terminal):
        self.storeData(s,a,r,ns,na)
        if self.samples_count == self.sample_window:
            STATS               = [] 
            start_time          = time()
            self.samples_count  = 0
            re_iteration        = 1 # Representation expansion iteration. Only used if the representation can be expanded 
            added_feature       = True
            while added_feature and re_iteration <= self.re_iterations:
                # Evaluate the policy and the corresponding PE-Error (Policy Evaluation) and TD-Error
                all_phi_s, PE_error, td_errors  = self.evaluatePolicy()
                # Save stats
                STATS.append([re_iteration,                     # iteration number
                              self.representation.features_num, # Number of features
                              PE_error,                         # Policy Evaluation Error
                              deltaT(start_time)                # Time since start
                              ])
                
                if not hasFunction(self.representation,'batchDiscover'):
                    break
                self.logger.log('Representation Expansion iteration #%d\n-----------------' % re_iteration)
                added_feature = self.representation.batchDiscover(td_errors, all_phi_s, self.data_s)
                re_iteration += 1
            self.STATS = array(STATS).T # Experiment will save this later
    def evaluatePolicy(self):
            #Calculate the Q for all samples using the new theta from LSTD
            #1. newTheta = LSTD
            #2. build phi_s_a for all samples
            #3. Q=phi*theta
            #4. Calculate ||Q-Q_MC||
            # returns all_phi_s (for samples used for LSTD) and td_erros (on samples used for LSTD)
            
            A,b, all_phi_s, all_phi_s_a, all_phi_ns, all_phi_ns_na = self.LSTD()
            p           = self.S.shape[0]
            n           = self.representation.features_num
            test_phi_s   = empty((p,n),dtype=self.representation.featureType())
            for i in arange(p):
                test_phi_s[i,:]  = self.representation.phi(self.S[i])
            all_test_phi_s_a  = self.representation.batchPhi_s_a(test_phi_s, self.A,use_sparse=self.use_sparse)
            Q           = all_test_phi_s_a * self.representation.theta.reshape(-1,1) if self.use_sparse else dot(all_test_phi_s_a,self.representation.theta)
            PE_error    = linalg.norm(Q.ravel()-self.Q_MC)
            self.logger.log("||Delta V|| = %f" % PE_error)
            return all_phi_s, PE_error, self.calculateTDErrors(self.data_r,all_phi_s_a,all_phi_ns_na) 
