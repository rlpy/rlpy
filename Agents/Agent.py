"""Standard Control Agent"""

from Tools import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Agent(object):
    """The Agent receives observations from the Domain and performs actions to
    obtain some goal.

    The agent interacts with the Domain in discrete timesteps. Each timestep,
    the %Agent receives some observations from the Domain and uses this
    information to update its Representation of the Domain.
    It then uses its Policy to select an action to perform.
    This process (observe, update, act) repeatsnuntil some goal or fail state,
    determined by the Domain, is reached. At this point the Experiment determines
    whether the agent starts over or has its current policy tested
    (without any exploration).

    The Agent class is a superclass that provides the basic framework for all RL agents.
    It provides the methods and attributes that allow child classes to interact
    with the Domain, Representation, Policy, and Experiment classes within the
    RLPy library.

    All new agent implementations should inherit from this class.
    """
    #: The Representation to be used by the %Agent
    representation      = None
    #: The domain the agent interacts with
    domain              = None
    #: The policy to be used by the agent
    policy              = None
    #: The initial learning rate. Note that initial_alpha should be set to 1 for automatic learning rate; otherwise, initial_alpha will act as a permanent upper-bound on alpha.
    initial_alpha       = 0.1
    #: The learning rate
    alpha               = 0
    #: The Candid Learning Rate. This value is updated in the updateAlpha method. We use the rate calculated by [Dabney W 2012] \n http://people.cs.umass.edu/~wdabney/papers/alphaBounds.p
    candid_alpha        = 0
    #: The eligibility trace, which marks states as eligible for a learning update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the parameter lambda is set. See: \n http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace   = []
    #: A simple object that records the prints in a file
    logger              = None
    #: Used by some alpha_decay modes
    episode_count       = 0
    #: Decay mode of learning rate. Options are determined by valid_decay_modes.
    alpha_decay_mode    = 'dabney'

    """Decay modes with an implementation.
    At the moment the choice is either 'boyan' or 'dabney'. Please use all lowercase letters to properly select a mode. \n
    The decay mode 'dabney' is an automatic rate described by Dabney. [Dabney W. 2012]"""
    valid_decay_modes   = ['dabney','boyan','const','boyan_const']
    #:  The N0 parameter for boyan learning rate decay
    boyan_N0            = 1000

    def __init__(self,representation,policy,domain,logger,initial_alpha = 0.1,alpha_decay_mode = 'dabney', boyan_N0 = 1000):
        self.representation = representation
        self.policy     = policy
        self.domain     = domain
        self.logger     = logger
        self.initial_alpha = initial_alpha
        self.alpha      = initial_alpha
        self.alpha_decay_mode = alpha_decay_mode.lower()
        self.boyan_N0   = boyan_N0
        if self.logger:
            self.logger.line()
            self.logger.log("Agent:\t\t"+str(className(self)))
            self.logger.log("Policy:\t\t"+str(className(self.policy)))
            if self.alpha_decay_mode == 'boyan': self.logger.log("boyan_N0:\t%.1f"%self.boyan_N0)
        # Check to make sure selected alpha_decay mode is valid
        if not self.alpha_decay_mode in self.valid_decay_modes:
            errMsg = "Invalid decay mode selected:"+self.alpha_decay -_mode+".\nValid choices are: "+str(self.valid_decay_modes)
            if self.logger:
                self.logger.log(errMsg)
            else: shout(errMsg)
            sys.exit(1)
        # Note that initial_alpha should be set to 1 for automatic learning rate; otherwise,
        # initial_alpha will act as a permanent upper-bound on alpha.
        if self.alpha_decay_mode == 'dabney':
            self.initial_alpha = 1.0
            self.alpha = 1.0

    def learn(self,s,p_actions, a,r,ns, np_actions, na,terminal):
        """
        This function gets observations of a transition and learns from it.
        Each inheriting class must implement this method.

        @param s: original state
        @param p_actions: possible actions in the original state
        @param a: action taken
        @param r: obtained reward
        @param ns: next state
        @oaram np_actions: possible actions in the next state
        @param na: action taken in the next state
        @param terminal: boolean indicating whether next state is terminal
        """
        return NotImplementedError


    def updateAlpha(self,phi_s, phi_prime_s, eligibility_trace_s, gamma, nnz, terminal):
        """Computes a new alpha for the agent based on self.alpha_decay_mode.

        @param nnz: The number of nonzero features
        @param terminal: Boolean that determines if the step is terminal or not
        @param phi_s: The feature vector evaluated at state (s)
        @param phi_prime_s: The feature vector evaluated at the new state (ns) = (s')
        @param gamma: The discount factor for learning
        @param eligibility_trace_s: Eligibility trace using state only (no copy-paste)
        """

        if self.alpha_decay_mode == 'dabney':
        # We only update alpha if this step is non-terminal; else phi_prime becomes
        # zero and the dot product below becomes very large, creating a very small alpha
            if not terminal:
                #Automatic learning rate: [Dabney W. 2012]
                self.candid_alpha    = abs(dot(gamma*phi_prime_s-phi_s,eligibility_trace_s)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.p
                self.candid_alpha    = 1/(self.candid_alpha*1.) if self.candid_alpha != 0 else inf
                self.alpha          = min(self.alpha,self.candid_alpha)
            # else we take no action
        elif self.alpha_decay_mode == 'boyan':
            self.alpha = self.initial_alpha * (self.boyan_N0 + 1.) / (self.boyan_N0 + (self.episode_count+1) ** 1.1) #New little change from not having +1 for episode count
            self.alpha /= sum(abs(phi_s)) # divide by l1 of the features; note that this method is only called if phi_s != 0
        elif self.alpha_decay_mode == 'boyan_const':
            self.alpha = self.initial_alpha * (self.boyan_N0 + 1.) / (self.boyan_N0 + (self.episode_count+1) ** 1.1) #New little change from not having +1 for episode count
        elif self.alpha_decay_mode == "const":
            self.alpha = self.initial_alpha
        else:
            shout("Unrecognized decay mode")
            self.logger.log("Unrecognized decay mode ")

    ## Run a single monte-carlo simulation episode from state s with action a following the current policy of the agent.
    # See code \ref Agent_MC_episode "Here".
    # @param s
    # The state used in the simulation
    # @param a
    # The action used in the simulation
    # @param tolerance
    # If the tolerance is set to a non-zero value, episodes will be stopped once the additive value to the sum of rewards drops below this threshold
    # @return eps_return
    # Sum of rewards
    # @return eps_length
    # Length of the Episode
    # @return eps_term
    # Specifies the terminal condition of the episode: 0 (stopped due to length), >0 (stopped due to a terminal state)
    # @return eps_discounted_return
    # Sum of discounted rewards.

    # [MC_episode code]
    def MC_episode(self,s=None,a=None, tolerance = 0):
        eps_length              = 0
        eps_return              = 0
        eps_discounted_return   = 0
        eps_term                = 0
        if s is None: s = self.domain.s0()
        if a is None: a = self.policy.pi(s)
        while not eps_term and eps_length < self.domain.episodeCap:
            r,ns,eps_term       = self.domain.step(a)
            s                   = ns
            eps_return          += r
            eps_discounted_return += self.representation.domain.gamma**eps_length * r
            eps_length          += 1
            a                   = self.policy.pi(s)
            if self.representation.domain.gamma**eps_length < tolerance:
                break
        return eps_return, eps_length, eps_term, eps_discounted_return
    # [MC_episode code]


    ## Use Monte-Carlo samples with the fixed policy to evaluate the Q(s,a).
    # See code \ref Agent_Q_MC "Here".
    # @param s
    # The state used in the simulation
    # @param a
    # The action used in the simulation
    # @param tolerance
    # If the tolerance is set to a non-zero value, episodes will be stopped once the additive value to the sum of rewards drops below this threshold
    # @param MC_samples
    # Number of samples to be used to evaluated the Q value
    # @return Q_avg
    # Averaged sum of discounted rewards = estimate of the Q

    # [Q_MC code]
    def Q_MC(self,s,a,MC_samples = 1000, tolerance = 0):
        Q_avg = 0
        for i in arange(MC_samples):
            #print "MC Sample:", i
            _,_,_,Q = self.MC_episode(s,a,tolerance)
            Q_avg = incrementalAverageUpdate(Q_avg,Q,i+1)
        return Q_avg
    # [Q_MC code]


    ## Evaluate the current policy for fixed number of samples and store them in samples-by-|S|+2.
    # Note: (2 corresponds to action and Q(s,a)) \n
    # Saves the data generated in a file. \n
    # See code \ref Agent_eval "Here".
    # @param samples
    # The number of samples (s,a)
    # @param MC_samples
    # The number of MC simulations used to estimate Q(s,a)
    # @param output_file
    # The file in which the data is saved.
    # The number of MC simulations used to estimate Q(s,a)
    # @return DATA
    # The data generated and stored in the output_file

    # [eval code]
    def evaluate(self,samples, MC_samples, output_file):
        tolerance       = 1e-10 #if gamma^steps falls bellow this number the MC-Chain will terminate since it will not have much impact in evaluation of Q
        cols            = self.domain.state_space_dims + 2
        DATA            = empty((samples,cols))
        terminal        = True
        steps           = 0
        while steps < samples:
            s = self.domain.s0() if terminal or steps % self.domain.episodeCap == 0 else s
            a = self.policy.pi(s)

            #Store the corresponding Q
            Q = self.Q_MC(s,a,MC_samples, tolerance)
            DATA[steps,:] = hstack((s,[a, Q]))
            r,s,terminal = self.domain.step(a)
            steps += 1

            self.logger.log("Sample "+ str(steps)+":"+ str(s)+" "+str(a)+" "+str(Q))

        save(output_file, DATA)
        return DATA
        # [eval code]

    def episodeTerminated(self):
        """This function adjusts all necessary elements of the agent at the end of
        the episodes.
        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.
        """
        #Increase the number of episodes
        self.episode_count += 1

        #Update the eligibility traces if they exist:
        # Set eligibility Traces to zero if it is end of the episode
        if hasattr(self, 'eligibility_trace'):
            if self.lambda_:
                self.eligibility_trace  = zeros(self.representation.features_num*self.domain.actions_num)
                self.eligibility_trace_s = zeros(self.representation.features_num)

