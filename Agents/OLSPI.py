######################################################
# Developed by Alborz Geramiard Feb 5th 2013 at MIT #
######################################################
# Online Least-Squares Policy Iteration
# The difference with LSPI is that OLSPI
# measures the performance of the policy on each iteration of LSPI
# using a single trajectory. At the end of iterations it selects the best iteration.
# It also adds the simulation to the pool of data
# use in the next iteration of LSPI. If the data cap is reached.
# If the data cap is reached it wont simulate more data and the last iteration
# is returned at the best policy.
 
 
from LSPI import *
class OLSPI(LSPI):
    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3, sample_cap = 200):
        self.sample_cap         = sample_cap
        super(OLSPI, self).__init__(representation,policy,domain,logger, lspi_iterations, sample_window, epsilon)
        if logger:
                logger.log('Sample Cap=\t%d' % sample_cap)
    def learn(self,s,a,r,ns,na,terminal):
        self.storeData(s,a,r,ns,na)        
        if self.samples_count == self.sample_window: #zero based hence the -1
            # Run LSTD for first solution
            A,b, all_phi_s, all_phi_s_a, all_phi_ns = self.LSTD()
            # Run Policy Iteration to change a_prime and recalculate theta
            self.policyIterationWithOnlineEvaulation(b,all_phi_s_a, all_phi_ns)
    def policyIterationWithOnlineEvaulation(self,b,all_phi_s_a,all_phi_ns):
            #Update the policy by recalculating A based on new na
            #Returns the TD error for each sample based on the latest weights and na
            # b is passed because it remains unchanged.
            phi_sa_size     = self.domain.actions_num*self.representation.features_num
            gamma           = self.domain.gamma
            td_errors       = empty((self.sample_window)) # holds the TD_errors for all samples

            #Begin updating the policy in LSPI loop
            weight_diff     = self.epsilon + 1 # So that the loop starts
            lspi_iteration  = 0
            self.logger.log('Running LSPI:')
            while lspi_iteration < self.lspi_iterations and weight_diff > self.epsilon:
                if phi_sa_size != 0: A = sp.coo_matrix((phi_sa_size,phi_sa_size))
                for i in arange(self.sample_window):
                    ns              = self.data_ns[i,:]
                    if phi_sa_size != 0:
                        phi_s_a         = all_phi_s_a[i,:]
                        phi_ns          = all_phi_ns[i,:]
                        new_na          = self.representation.bestAction(ns,phi_ns)
                        phi_ns_new_na   = self.representation.phi_sa(ns,new_na,phi_ns)
                        d               = phi_s_a-gamma*phi_ns_new_na
                        A               = A + outer(phi_s_a,d)
                        td_errors[i]    = self.data_r[i]+dot(-d,self.representation.theta)
                    else:
                        td_errors[i]    = self.data_r[i]
                #Calculate theta
                if phi_sa_size != 0:
                    new_theta                   = solveLinear(sp.csc_matrix(A),b)
                    weight_diff                 = linalg.norm(self.representation.theta - new_theta)
                    if weight_diff > self.epsilon: self.representation.theta   = new_theta
                    self.logger.log("%d: L2_norm of weight difference = %0.3f, Density of A: %0.2f%%" % (lspi_iteration+1,weight_diff, count_nonzero(A)/(prod(A.shape)*1.)*100))
                else:
                    print "No features, hence no more iterations is necessary!"
                    weight_diff = 0
                lspi_iteration +=1
            return td_errors
    def LSTD(self): 
        
        #No features means empty matrices
        if self.representation.features_num == 0:
            return array([]), array([]), array([]), array([]), array([])
         
        self.samples_count  = 0
        # Calculate the A and b matrixes in LSTD
        phi_sa_size     = self.domain.actions_num*self.representation.features_num
        A               = sp.coo_matrix((phi_sa_size,phi_sa_size))
        b               = zeros(phi_sa_size)
        all_phi_s       = zeros((self.sample_window,self.representation.features_num)) #phi_s will be saved for batch iFDD
        all_phi_s_a     = zeros((self.sample_window,phi_sa_size)) #phi_sa will be fixed during iterations
        all_phi_ns      = zeros((self.sample_window,self.representation.features_num)) #phi_ns_na will change according to na so we only cache the phi_na which remains the same
        
        #print "Making A,b"
        gamma               = self.representation.domain.gamma
        for i in arange(self.sample_window):
            s                   = self.data_s[i]
            ns                  = self.data_ns[i]
            a                   = self.data_a[i]
            na                  = self.data_na[i]
            r                   = self.data_r[i]
            phi_s               = self.representation.phi(s)
            phi_s_a             = self.representation.phi_sa(s,a,phi_s)
            phi_ns              = self.representation.phi(ns)
            phi_ns_na           = self.representation.phi_sa(ns,na,phi_ns)
            all_phi_s[i,:]      = phi_s
            all_phi_s_a[i,:]    = phi_s_a
            all_phi_ns[i,:]     = phi_ns
            d                   = phi_s_a-gamma*phi_ns_na
            A                   = A + outer(phi_s_a,d) 
            b                   = b + r*phi_s_a

        #Calculate theta
        self.representation.theta = solveLinear(sp.csc_matrix(A),b)
        #Calculate TD-Error
        return A,b, all_phi_s, all_phi_s_a, all_phi_ns
    def storeData(self,s,a,r,ns,na):
        self.data_s[self.samples_count,:]   = s
        self.data_a[self.samples_count]   = a
        self.data_r[self.samples_count]   = r
        self.data_ns[self.samples_count,:]  = ns
        self.data_na[self.samples_count]  = na
        self.samples_count                  += 1
