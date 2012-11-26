######################################################
# Developed by Alborz Geramiard Nov 19th 2012 at MIT #
######################################################
# Least-Squares Policy Iteration [Lagoudakis and Parr 2003]
# This version recalculates the policy every <sample_window>. Samples are obtained using the recent version of the policy  
from OnlineAgent import *
class LSPI(OnlineAgent):
    lspi_iterations = 0         # Number of LSPI iterations
    sample_window   = 0         # Number of samples to be used to calculate the LSTD solution
    samples_count   = 0         # Counter for the sample count
    epsilon         = 0         # Minimum l_2 change required to continue iterations in LSPI
    #Store Data in separate matrixes
    data_s          = []        # 
    data_a          = []        # 
    data_r          = []        #
    data_ns         = []        # 
    data_na         = []        # 
        
    def __init__(self,representation,policy,domain, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3):
        self.samples_count      = 0
        self.sample_window      = sample_window
        self.epsilon            = epsilon
        self.lspi_iterations    = lspi_iterations
        self.phi_sa_size        = domain.actions_num * representation.features_num
        self.data_s             = zeros((sample_window, domain.state_space_dims))
        self.data_ns            = zeros((sample_window, domain.state_space_dims))
        self.data_a             = zeros((sample_window, 1))
        self.data_na            = zeros((sample_window, 1))
        self.data_r             = zeros((sample_window, 1))
        super(LSPI, self).__init__(representation, policy, domain)
    def learn(self,s,a,r,ns,na,terminal):
        
        self.storeData(s,a,r,ns,na)        
        if self.samples_count == self.sample_window: #zero based hence the -1
            self.samples_count  = 0
            # Calculate the A and b matrixes in LSTD
            phi_sa_size = self.domain.actions_num*self.representation.features_num
            A           = zeros((phi_sa_size,phi_sa_size))
            b           = zeros(phi_sa_size)
            all_phi_s   = zeros((self.sample_window,self.representation.features_num)) #phi_s will be saved for batch iFDD
            all_phi_s_a = zeros((self.sample_window,phi_sa_size)) #phi_sa will be fixed during iterations
            all_phi_ns  = zeros((self.sample_window,self.representation.features_num)) #phi_ns_na will change according to na so we only cache the phi_na which remains the same
            
            #print "Making A,b"
            for i in range(self.sample_window):
                gamma               = self.representation.domain.gamma
                s                   = self.data_s[i]
                ns                  = self.data_ns[i]
                a                   = self.data_a[i]
                na                  = self.data_na[i]
                r                   = self.data_r[i]
                phi_s               = self.representation.phi(s)
                phi_s_a             = self.representation.phi_sa_from_phi_s(phi_s,a)
                phi_ns              = self.representation.phi(ns)
                phi_ns_na           = self.representation.phi_sa_from_phi_s(phi_ns,na)
                all_phi_s[i,:]      = phi_s
                all_phi_s_a[i,:]    = phi_s_a
                all_phi_ns[i,:]     = phi_ns
                d                   = phi_s_a-gamma*phi_ns_na
                A                   += outer(phi_s_a,d) #this is because phi_s_a is 1-by-n instead of n-by-1
                b                   += r*phi_s_a

            #Calculate theta
            #print "inverting"
            self.representation.theta = solveLinear(A,b)
            
            #Begin updating the policy in LSPI loop
            weight_diff     = self.epsilon + 1 # So that the loop starts
            lspi_iteration  = 0
            while lspi_iteration < self.lspi_iterations and weight_diff > self.epsilon:
                A = zeros((phi_sa_size,phi_sa_size))
                for i in range(self.sample_window):
                    phi_s_a         = all_phi_s_a[i,:]
                    phi_ns          = all_phi_ns[i,:]
                    ns              = self.data_ns[i]
                    new_na          = self.representation.bestAction(ns)
                    phi_ns_new_na   = self.representation.phi_sa_from_phi_s(phi_ns,new_na)
                    d               = phi_s_a-gamma*phi_ns_new_na
                    A               += outer(phi_s_a,d) #this is because phi_s_a is 1-by-n instead of n-by-1
                    
                #Calculate theta
                new_theta                   = solveLinear(A,b)
                weight_diff                 = linalg.norm(self.representation.theta - new_theta)
                self.representation.theta   = new_theta
                print "%d: L2_norm of weight difference = %0.3f" % (lspi_iteration,weight_diff)
                lspi_iteration +=1
    def storeData(self,s,a,r,ns,na):
        self.data_s[self.samples_count,:]   = s
        self.data_a[self.samples_count,:]   = a
        self.data_r[self.samples_count,:]   = r
        self.data_ns[self.samples_count,:]  = ns
        self.data_na[self.samples_count,:]  = na
        self.samples_count                  += 1
