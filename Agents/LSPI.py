######################################################
# Developed by Alborz Geramiard Nov 19th 2012 at MIT #
######################################################
# Least-Squares Policy Iteration [Lagoudakis and Parr 2003]
# This version recalculates the policy every <sample_window>. Samples are obtained using the recent version of the policy  
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from Agent import *
from Domains import *
class LSPI(Agent):
    use_sparse      = 1         # Use sparse representation for A?
    lspi_iterations = 0         # Number of LSPI iterations
    sample_window   = 0         # Number of samples to be used to calculate the LSTD solution
    samples_count   = 0         # Counter for the sample count
    epsilon         = 0         # Minimum l_2 change required to continue iterations in LSPI
    
    return_best_policy  = 0        # If this flag is activated, on each iteration of LSPI, the policy is checked with one simulation and in the end the theta w.r.t the best policy is returned. Note that this will require more samples than the intial set of samples provided to LSPI
    best_performance    = -inf     # In the "return_best_policy", The best perofrmance check is stored here through LSPI iterations
    best_theta          = None     # In the "return_best_policy", The best theta is stored here through LSPI iterations
    best_TD_errors       = None     # In the "return_best_policy", The TD_Error corresponding to the best theta is stored here through LSPI iterations 
    extra_samples       = 0        # The number of extra samples used due to extra simulations 
    #Store Data in separate matrixes
    data_s          = []        # 
    data_a          = []        # 
    data_r          = []        #
    data_ns         = []        # 
    data_na         = []        # 
    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3,return_best_policy = 0):
        self.samples_count      = 0
        self.sample_window      = sample_window
        self.epsilon            = epsilon
        self.lspi_iterations    = lspi_iterations
        self.return_best_policy = return_best_policy # Default is False. If set True it will track the best policy during iterations
        self.phi_sa_size        = domain.actions_num * representation.features_num
        self.data_s             = zeros((sample_window, domain.state_space_dims))
        self.data_ns            = zeros((sample_window, domain.state_space_dims))
        self.data_a             = zeros((sample_window,1),dtype=uint16)
        self.data_na            = zeros((sample_window,1),dtype=uint16)
        self.data_r             = zeros((sample_window,1))
        super(LSPI, self).__init__(representation, policy, domain,logger)
        if logger:
                self.logger.log('Max LSPI Iterations:\t%d' % lspi_iterations)
                self.logger.log('Data Size:\t\t%d' % sample_window)
                self.logger.log('Weight Difference tol.:\t%0.3f' % epsilon)
                self.logger.log('Track the best policy:\t%d' % self.return_best_policy)
    def learn(self,s,a,r,ns,na,terminal):
        
        self.storeData(s,a,r,ns,na)        
        if self.samples_count == self.sample_window: #zero based hence the -1
            self.samples_count = 0
            # Run LSTD for first solution
            A,b,all_phi_s, all_phi_s_a, all_phi_ns = self.LSTD()
            # Run Policy Iteration to change a_prime and recalculate theta
            self.policyIteration(b,all_phi_s_a, all_phi_ns)
    def policyIteration(self,b,all_phi_s_a,all_phi_ns):
            # Update the policy by recalculating A based on new na
            # Returns the TD error for each sample based on the latest weights and next actions
            # b is passed as an input because it remains unchanged during policy iteration.
            start_time = time()
            if all_phi_ns.shape[0] == 0:
                print "No features, hence no more iterations is necessary!"
                weight_diff = 0
                return []
            
            #Begin updating the policy in LSPI loop
            weight_diff     = self.epsilon + 1 # So that the loop starts
            lspi_iteration  = 0
            self.best_performance = -inf
            self.logger.log('Running Policy Iteration:')
            
            # We save action_mask on the first iteration (used for batchBestAction) to reuse it and boost the speed
            # action mask is a matrix that shows which actions are available for each state
            action_mask = None 
            F1          = sp.csr_matrix(all_phi_s_a) if self.use_sparse else all_phi_s_a
            R           = self.data_r
            W           = self.representation.theta
            gamma       = self.domain.gamma
            while lspi_iteration < self.lspi_iterations and weight_diff > self.epsilon:
                
                #Find the best action for each state given the current value function
                #Notice if actions have the same value the first action is selected in the batch mode
                iteration_start_time = time()
                bestAction, all_phi_ns_na,action_mask = self.representation.batchBestAction(self.data_ns,all_phi_ns,action_mask,self.use_sparse)
                #Recalculate A matrix (b remains the same)
                if self.use_sparse:
                    F2  = sp.csr_matrix(all_phi_ns_na)
                    A   = F1.T*(F1 - gamma*F2)
                else:
                    F2  = all_phi_ns_na
                    A   = dot(F1.T, F1 - gamma*F2)
                
                A                           = regularize(A)
                #Solve for the new weight
                td_errors                   = (R+(gamma*F2-F1)*self.representation.theta.reshape(-1,1)).ravel() if self.use_sparse else R+dot(gamma*F2-F1,self.representation.theta)
                new_theta, solve_time       = solveLinear(A,b)
                if self.return_best_policy:
                    self.updateBestPolicy(new_theta,td_errors)
                else:
                    eps_return, eps_length, _   = self.checkPerformance(); self.logger.log(">>> %+0.3f Return, %d Steps, %d Features" % (eps_return, eps_length, self.representation.features_num))

                weight_diff = linalg.norm(self.representation.theta - new_theta)
                if weight_diff > self.epsilon: 
                    self.representation.theta = new_theta
                    if solve_time > 1: #log solve time only if takes more than 1 second
                        self.logger.log("%d: ||w1-w2|| = %0.3f, Sparsity: %0.1f%%, Iteration in %0.0f(s), Solved in %0.0f(s)" % (lspi_iteration+1,weight_diff, sparsity(A),deltaT(iteration_start_time),solve_time))
                    else:
                        self.logger.log("%d: ||w1-w2|| = %0.3f, Sparsity: %0.1f%%, Iteration in %0.0f(s)" % (lspi_iteration+1,weight_diff, sparsity(A),deltaT(iteration_start_time)))
                lspi_iteration +=1
            
            self.logger.log('Total Policy Iteration Time = %0.0f(s)' % deltaT(start_time))
            if self.return_best_policy: 
                self.logger.log("%d Extra Samples So Far." % self.extra_samples)
                self.representation.theta = self.best_theta
                return self.best_TD_errors
            else:
                return td_errors
    def policyIteration_non_matrix_version(self,b,all_phi_s_a,all_phi_ns):
            # Update the policy by recalculating A based on new na
            # Returns the TD error for each sample based on the latest weights and next actions
            # b is passed as an input because it remains unchanged during policy iteration.
            phi_sa_size     = self.domain.actions_num*self.representation.features_num
            gamma           = self.domain.gamma
            td_errors       = empty((self.sample_window)) # holds the TD_errors for all samples
            #Begin updating the policy in LSPI loop
            weight_diff     = self.epsilon + 1 # So that the loop starts
            lspi_iteration  = 0
            self.best_performance = -inf
            self.logger.log('Running Policy Iteration:')
            while lspi_iteration < self.lspi_iterations and weight_diff > self.epsilon:
                iteration_start_time = time()
                if phi_sa_size != 0: 
                    if self.use_sparse:
                        A = sp.csr_matrix((phi_sa_size,phi_sa_size)) # Reset the A matrix
                    else:
                        A = zeros((phi_sa_size,phi_sa_size))
                for i in arange(self.sample_window):
                    ns              = self.data_ns[i,:]
                    if phi_sa_size != 0:
                        phi_ns          = all_phi_ns[i,:]
                        new_na          = self.representation.bestAction(ns,phi_ns)
                        phi_s_a         = all_phi_s_a[i,:]
                        phi_ns_new_na   = self.representation.phi_sa(ns,new_na,phi_ns)
                        if self.use_sparse:
                            phi_s_a         = sp.csr_matrix(phi_s_a,dtype=all_phi_s_a.dtype)
                            phi_ns_new_na   = sp.csr_matrix(phi_ns_new_na,dtype=all_phi_s_a.dtype)
                            d               = phi_s_a-gamma*phi_ns_new_na
                            A               = A + phi_s_a.T*d
                            td_errors[i]    = self.data_r[i]+sp_dot_array(-d,self.representation.theta)
                        else:
                            d               = phi_s_a-gamma*phi_ns_new_na
                            A               = A + outer(phi_s_a,d)
                            td_errors[i]    = self.data_r[i]+dot(-d,self.representation.theta)
                    else:
                        td_errors[i]    = self.data_r[i]
                #Calculate theta
                if phi_sa_size != 0:
                    #Regularaize A
                    A                           = regularize(A)
                    new_theta, solve_time       = solveLinear(A,b)
                    weight_diff                 = linalg.norm(self.representation.theta - new_theta)
                    if self.return_best_policy:
                        # Check Performance with new theta
                        old_theta                   = array(self.representation.theta)
                        self.representation.theta   = new_theta
                        eps_return, eps_length, _   = self.checkPerformance(); self.logger.log(">>> %+0.3f Return, %d Steps, %d Features" % (eps_return, eps_length, self.representation.features_num))
                        self.extra_samples          += eps_length
                        performance                 = eps_length if isinstance(self.representation.domain,Pendulum_InvertedBalance) else eps_return
                        if self.best_performance < performance:
                            self.best_performance   = performance
                            self.best_TD_errors     = td_errors
                            self.best_theta         = array(new_theta)
                            self.logger.log('[Saved]')
                        self.representation.theta = old_theta #Return to previous theta
                    if weight_diff > self.epsilon: 
                        self.representation.theta   = new_theta
                        eps_return, eps_length, _   = self.checkPerformance(); self.logger.log(">>> %+0.3f Return, %d Steps, %d Features" % (eps_return, eps_length, self.representation.features_num))
                    if solve_time > 1: #log solve time only if takes more than 1 second
                        self.logger.log("%d: ||w1-w2|| = %0.3f, Sparsity: %0.1f%%, Iteration in %0.0f(s), Solved in %0.0f(s)" % (lspi_iteration+1,weight_diff, sparsity(A),deltaT(iteration_start_time),solve_time))
                    else:
                        self.logger.log("%d: ||w1-w2|| = %0.3f, Sparsity: %0.1f%%, Iteration in %0.0f(s)" % (lspi_iteration+1,weight_diff, sparsity(A),deltaT(iteration_start_time)))
                else:
                    self.logger.log("No features, hence no more iterations is necessary!")
                    weight_diff = 0
                lspi_iteration +=1
            if self.return_best_policy: 
                self.logger.log("%d Extra Samples So Far." % self.extra_samples)
                self.representation.theta = self.best_theta
                return self.best_TD_errors
            else:
                return td_errors
    def updateBestPolicy(self,new_theta,new_td_error):
        # Check the performance of the policy corresponding to the new_theta
        # Logs the best found theta, performance, and td_error based on a single run of the  new theta
        old_theta                   = array(self.representation.theta)
        self.representation.theta   = new_theta
        eps_return, eps_length, _   = self.checkPerformance(); self.logger.log(">>> %+0.3f Return, %d Steps, %d Features" % (eps_return, eps_length, self.representation.features_num))
        self.extra_samples          += eps_length
        performance                 = eps_length if isinstance(self.representation.domain,Pendulum_InvertedBalance) else eps_return
        if self.best_performance < performance:
            self.best_performance   = performance
            self.best_TD_errors     = new_td_error
            self.best_theta         = array(new_theta)
            self.logger.log('[Saved]')
            self.representation.theta = old_theta #Return to previous theta
    def LSTD(self): 
        # If sameSamples = True then LSTD will use existing all_phi_s, all_phi_s_a, and all_phi_ns 
        start_time = time()
        if self.sample_window == 0:
            print 'Window Size for LSPI should not be 0!'
            return None

        self.logger.log('Running LSTD:')
        #No features means empty matrices
        if self.representation.features_num == 0:
            return array([]), array([])
         
        
        #build phi_s and phi_ns for all samples
        p           = self.data_s.shape[0]
        n           = self.representation.features_num
        all_phi_s   = empty((p,n),dtype=self.representation.featureType())
        all_phi_ns  = empty((p,n),dtype=self.representation.featureType())
        for i in arange(self.sample_window):
            all_phi_s[i,:]  = self.representation.phi(self.data_s[i])
            all_phi_ns[i,:] = self.representation.phi(self.data_ns[i])
            
        #build phi_s_a and phi_ns_na for all samples given phi_s and phi_ns
        all_phi_s_a     = self.representation.batchPhi_s_a(all_phi_s, self.data_a,use_sparse=self.use_sparse)
        all_phi_ns_na   = self.representation.batchPhi_s_a(all_phi_ns, self.data_na,use_sparse=self.use_sparse)
        
        #calculate A and b for LSTD
        F1              = all_phi_s_a
        F2              = all_phi_ns_na
        R               = self.data_r
        gamma           = self.domain.gamma
        
        b = dot(F1.T,R).reshape(-1,1)
        A = dot(F1.T, F1 - gamma*F2)
        A = regularize(A)
        #Calculate theta
        self.representation.theta, solve_time  = solveLinear(A,b)
        if solve_time > 1: #log solve time only if takes more than 1 second
            self.logger.log('Total LSTD Time = %0.0f(s), Solve Time = %0.0f(s)' % (deltaT(start_time), solve_time))
        else:   
            self.logger.log('Total LSTD Time = %0.0f(s)' % (deltaT(start_time)))
        return A,b, all_phi_s, all_phi_s_a, all_phi_ns
    def LSTD_non_matrix_version(self): 
        start_time = time()
        if self.sample_window == 0:
            print 'Window Size for LSPI should not be 0!'
            return 

        self.logger.log('Running LSTD:')
        #No features means empty matrices
        if self.representation.features_num == 0:
            return array([]), array([]), array([]), array([]), array([])
         
        self.samples_count  = 0
       
        # Make one sample phi to decide about the type of A matrix:
        phi_s               = self.representation.phi(self.data_s[0])

        # Calculate the A and b matrixes in LSTD
        phi_sa_size     = self.domain.actions_num*self.representation.features_num
        if self.use_sparse:
            A               = sp.csr_matrix((phi_sa_size,phi_sa_size)) # A matrix is in general float
        else:
            A               = zeros((phi_sa_size,phi_sa_size)) # A matrix is in general float
        b               = zeros(phi_sa_size)
        all_phi_s       = zeros((self.sample_window,self.representation.features_num),dtype=phi_s.dtype) #phi_s will be saved for batch iFDD
        all_phi_s_a     = zeros((self.sample_window,phi_sa_size),dtype=phi_s.dtype) #phi_sa will be fixed during iterations
        all_phi_ns      = zeros((self.sample_window,self.representation.features_num),dtype=phi_s.dtype) #phi_ns_na will change according to na so we only cache the phi_na which remains the same
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
            b                   = b + r*phi_s_a
            if self.use_sparse:
                phi_s_a             = sp.csr_matrix(phi_s_a,dtype=phi_s_a.dtype)
                phi_ns_na           = sp.csr_matrix(phi_ns_na,dtype=phi_ns_na.dtype)
                d                   = phi_s_a-gamma*phi_ns_na
                A                   = A + phi_s_a.T*d
            else:
                d                   = phi_s_a-gamma*phi_ns_na
                A                   = A + outer(phi_s_a,d)
        
        #Regularaize A
        A = regularize(A)
        
        #Calculate theta
        self.representation.theta, solve_time  = solveLinear(A,b)
        if solve_time > 1: #log solve time only if takes more than 1 second
            self.logger.log('Total LSTD Time = %0.0f(s), Solve Time = %0.0f(s)' % (deltaT(start_time), solve_time))
        else:   
            self.logger.log('Total LSTD Time = %0.0f(s)' % (deltaT(start_time)))
        return A,b, all_phi_s, all_phi_s_a, all_phi_ns
    def storeData(self,s,a,r,ns,na):
        self.data_s[self.samples_count,:]   = s
        self.data_a[self.samples_count]     = a
        self.data_r[self.samples_count]     = r
        self.data_ns[self.samples_count,:]  = ns
        self.data_na[self.samples_count]    = na
        self.samples_count                  += 1
