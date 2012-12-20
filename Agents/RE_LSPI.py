######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################
# Least-Squares Policy Iteration [Lagoudakis and Parr 2003] Integrated with Representation Expansion Techniques
# LOOP:
# 1. Run LSPI
# 2. Run feature expansion 
from LSPI import *
class RE_LSPI(LSPI):
    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3, re_iterations = 100):
        assert isinstance(representation,iFDD)
        self.re_iterations = re_iterations # Number of iterations over LSPI and iFDD
        super(RE_LSPI, self).__init__(representation,policy,domain,logger,lspi_iterations, sample_window, epsilon)
        if logger:
            logger.log('Max Representation Expansion Iterations:\t%d' % re_iterations)
    def learn(self,s,a,r,ns,na,terminal):
        self.storeData(s,a,r,ns,na)        
        if self.samples_count == self.sample_window: #zero based hence the -1
            re_iteration = 1
            added_feature        = True
            while added_feature and re_iteration <= self.re_iterations:
                self.logger.log('Representation Expansion iteration #%d\n-----------------' % re_iteration)
                # Run LSTD for first solution
                A,b, all_phi_s, all_phi_s_a, all_phi_ns, td_errors = self.LSTD()
                # Run Policy Iteration to change a_prime and recalculate theta
                td_errors = self.policyIteration(b,all_phi_s_a,all_phi_ns)
                # Add new Features
                if hasFunction(self.representation,'batchDiscover'):
                    added_feature = self.representation.batchDiscover(td_errors, all_phi_s)
                else:
                    self.logger.log('%s does not have Batch Discovery!' % classname(self.representation))
                    added_features = 0
                re_iteration += 1