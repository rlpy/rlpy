"""Fifty state chain."""

from rlpy.Tools import plt, mpatches
import numpy as np
from .Domain import Domain

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Alborz Geramifard", "Robert H. Klein"]


class FiftyChain(Domain):

    """
    Random start location, goal is to proceed to nearest reward. \n
    **STATE:** s0 <-> s1 <-> ... <-> s49 \n
    **ACTIONS:** left [0] or right [1] \n
    Actions succeed with probability .9, otherwise execute opposite action. 
    Note that the actions [left, right] are available in ALL states, but if 
    left is selected in s0 or right in s49, then s remains unchanged. \n

    .. note::
        The optimal policy is to always go to the nearest goal

    **REWARD:** of +1 at states 10 and 41 (indices 9 and 40).  Reward is 
    obtained when transition out of the reward state, not when first enter. \n
    
    Note that this class provides the 
    function :py:meth`~rlpy.Domains.FiftyChain.L_inf_distance_to_V_star`, which 
    accepts an arbitrary representation and returns the error between it and 
    the optimal policy.
    The user can also enforce actions under the optimal policy (ignoring the 
    agent's policy) by setting ``using_optimal_policy=True`` in FiftyChain.py.

    **REFERENCE:**

    .. seealso::
        Michail G. Lagoudakis, Ronald Parr, and L. Bartlett
        Least-squares policy iteration.  Journal of Machine Learning Research
        (2003) Issue 4.
    """

    #: Reward for each timestep spent in the goal region
    GOAL_REWARD = 1
    #: Indices of states with rewards
    GOAL_STATES = [9, 40]
    #: Set by the domain = min(100,rows*cols)
    episodeCap = 50
    # Used for graphical normalization
    MAX_RETURN = 2.5
    # Used for graphical normalization
    MIN_RETURN = 0
    # Used for graphical shifting of arrows
    SHIFT = .01
    # Used for graphical radius of states
    RADIUS = .05
    # Stores the graphical paths for states so that we can later change their
    # colors
    circles = None
    #: Number of states in the chain
    chainSize = 50
    # Y values used for drawing circles
    Y = 1
    actions_num = 2
    #: Probability of taking the other (unselected) action
    p_action_failure = 0.1
    V_star = [
        0.25424059953210576,
        0.32237043339365301,
        0.41732244995544071,
        0.53798770416976416,
        0.69467264588670452,
        0.91307612341516964,
        1.1996067857970858,
        1.5914978718669359,
        2.1011316163482885,
        2.7509878207260079,
        2.2007902565808002,
        1.7606322052646419,
        1.4085057642117096,
        1.1268046113693631,
        0.90144368909548567,
        0.72115495127639073,
        0.5769239610211111,
        0.46153916881688833,
        0.36923133505350991,
        0.29538506804280829,
        0.23630805443424513,
        0.18904644354739669,
        0.15123715483791522,
        0.12098972387033219,
        0.096791779096267572,
        0.077433423277011526,
        0.064110827579671889,
        0.080577201155275072,
        0.10271844729124571,
        0.13354008685155827,
        0.17749076168535796,
        0.22641620289304287,
        0.29916005826456937,
        0.39326998437016564,
        0.52325275246999614,
        0.67438770340963006,
        0.90293435616054674,
        1.1704408409975584,
        1.5213965403184493,
        2.0462009513290296,
        2.7423074964894685,
        2.1938459971915725,
        1.7550767977532584,
        1.404061438202612,
        1.1232491505620894,
        0.89859932044966939,
        0.71887945635973116,
        0.57510356508778659,
        0.46008285207022837,
        0.36806628165617972]          # Array of optimal values at each state

    # The optimal policy for this domain
    optimalPolicy = None
    # Should the domain only allow optimal actions
    using_optimal_policy = False

    # Plotting values
    domain_fig = None
    value_function_fig = None
    policy_fig = None
    V_star_line = None
    V_approx_line = None
    COLORS = ['g', 'k']
    LEFT = 0
    RIGHT = 1

    # Constants in the map
    def __init__(self):
        self.start = 0
        self.statespace_limits = np.array([[0, self.chainSize - 1]])
        super(FiftyChain, self).__init__()
        # To catch errors
        self.optimal_policy = np.array([-1 for dummy in xrange(0, self.chainSize)])
        self.storeOptimalPolicy()
        self.discount_factor = 0.8  # Set discount_factor to be 0.8 for this domain per L & P 2007

    def storeOptimalPolicy(self):
        """
        Computes and stores the optimal policy on this particular chain.

        .. warning::

           This ONLY applies for the scenario where two states provide
           reward - this policy will be suboptimal for all other domains!

        """
        self.optimal_policy[np.arange(self.GOAL_STATES[0])] = self.RIGHT
        goalStateIndices = np.arange(1, len(self.GOAL_STATES))
        for i in goalStateIndices:
            goalState1 = self.GOAL_STATES[i - 1]
            goalState2 = self.GOAL_STATES[i]
            averageState = int(np.mean([goalState1, goalState2]))
            self.optimal_policy[np.arange(goalState1, averageState)] = self.LEFT
            self.optimal_policy[np.arange(averageState, goalState2)] = self.RIGHT
        self.optimal_policy[np.arange(
            self.GOAL_STATES[-1],
            self.chainSize)] = self.LEFT

    def showDomain(self, a=0):
        s = self.state
        # Draw the environment
        if self.circles is None:
            self.domain_fig = plt.subplot(3, 1, 1)
            plt.figure(1, (self.chainSize * 2 / 10.0, 2))
            self.domain_fig.set_xlim(0, self.chainSize * 2 / 10.0)
            self.domain_fig.set_ylim(0, 2)
            self.domain_fig.add_patch(
                mpatches.Circle((1 / 5.0 + 2 / 10.0 * (self.chainSize - 1),
                                 self.Y),
                                self.RADIUS * 1.1,
                                fc="w"))  # Make the last one double circle
            self.domain_fig.xaxis.set_visible(False)
            self.domain_fig.yaxis.set_visible(False)
            self.circles = [mpatches.Circle((1 / 5.0 + 2 / 10.0 * i, self.Y), self.RADIUS, fc="w")
                            for i in range(self.chainSize)]
            for i in range(self.chainSize):
                self.domain_fig.add_patch(self.circles[i])
                plt.show()
        for p in self.circles:
            p.set_facecolor('w')
        for p in self.GOAL_STATES:
            self.circles[p].set_facecolor('g')
        self.circles[s].set_facecolor('k')
        plt.draw()

    def showLearning(self, representation):
        allStates = np.arange(0, self.chainSize)
        X = np.arange(self.chainSize) * 2.0 / 10.0 - self.SHIFT
        Y = np.ones(self.chainSize) * self.Y
        DY = np.zeros(self.chainSize)
        DX = np.zeros(self.chainSize)
        C = np.zeros(self.chainSize)

        if self.value_function_fig is None:
            self.value_function_fig = plt.subplot(3, 1, 2)
            self.V_star_line = self.value_function_fig.plot(
                allStates,
                self.V_star)
            V = [representation.V(s, False, self.possibleActions(s=s))
                 for s in allStates]

            # Note the comma below, since a tuple of line objects is returned
            self.V_approx_line, = self.value_function_fig.plot(
                allStates, V, 'r-', linewidth=3)
            self.V_star_line = self.value_function_fig.plot(
                allStates,
                self.V_star,
                'b--',
                linewidth=3)
            # Maximum value function is sum of all possible rewards
            plt.ylim([0, self.GOAL_REWARD * (len(self.GOAL_STATES) + 1)])

            self.policy_fig = plt.subplot(3, 1, 3)
            self.policy_fig.set_xlim(0, self.chainSize * 2 / 10.0)
            self.policy_fig.set_ylim(0, 2)
            self.arrows = plt.quiver(
                X,
                Y,
                DX,
                DY,
                C,
                cmap='fiftyChainActions',
                units='x',
                width=0.05,
                scale=.008,
                alpha=.8)  # headwidth=.05, headlength = .03, headaxislength = .02)
            self.policy_fig.xaxis.set_visible(False)
            self.policy_fig.yaxis.set_visible(False)

        V = [representation.V(s, False, self.possibleActions(s=s))
             for s in allStates]
        pi = [representation.bestAction(s, False, self.possibleActions(s=s))
              for s in allStates]
        #pi  = [self.optimal_policy[s] for s in allStates]

        DX = [(2 * a - 1) * self.SHIFT * .1 for a in pi]

        self.V_approx_line.set_ydata(V)
        self.arrows.set_UVC(DX, DY, pi)
        plt.draw()

    def step(self, a):
        s = self.state
        actionFailure = (
            self.random_state.random_sample(
            ) < self.p_action_failure)
        if a == self.LEFT or (a == self.RIGHT and actionFailure):  # left
            ns = max(0, self.state - 1)
        elif a == self.RIGHT or (a == self.LEFT and actionFailure):
            ns = min(self.chainSize - 1, self.state + 1)
        self.state = ns
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if s in self.GOAL_STATES else 0
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.random_state.randint(0, self.chainSize)
        return self.state, self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        return False

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        if self.using_optimal_policy:
            return np.array([self.optimal_policy[s]])
        else:
            return np.arange(self.actions_num)

    def L_inf_distance_to_V_star(self, representation):
        """
        :param representation: An arbitrary learned representation of the
            value function.

        :return: the L-infinity distance between the parameter representation
            and the optimal one.
        """
        V = np.array([representation.V(s, False, self.possibleActions(s=s)) \
                      for s in xrange(self.chainSize)])
        return np.linalg.norm(V - self.V_star, np.inf)
