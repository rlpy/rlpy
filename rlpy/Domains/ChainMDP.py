"""Simple Chain MDP domain."""
from rlpy.Tools import plt, mpatches, fromAtoB
from .Domain import Domain
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class ChainMDP(Domain):

    """
    A simple Chain MDP.

    **STATE:** s0 <-> s1 <-> ... <-> sn \n
    **ACTIONS:** are left [0] and right [1], deterministic. \n
    
    .. note::
    
        The actions [left, right] are available in ALL states, but if 
        left is selected in s0 or right in sn, then s remains unchanged.

    The task is to reach sn from s0, after which the episode terminates.

    .. note::
        Optimal policy is to always to go right.

    **REWARD:**
    -1 per step, 0 at goal (terminates)

    **REFERENCE:**

    .. seealso::
        Michail G. Lagoudakis, Ronald Parr, and L. Bartlett
        Least-squares policy iteration.  Journal of Machine Learning Research
        (2003) Issue 4.
    """

    #: Reward for each timestep spent in the goal region
    GOAL_REWARD = 0
    #: Reward for each timestep
    STEP_REWARD = -1
    #: Set by the domain = min(100,rows*cols)
    episodeCap = 0
    # Used for graphical normalization
    MAX_RETURN = 1
    # Used for graphical normalization
    MIN_RETURN = 0
    # Used for graphical shifting of arrows
    SHIFT = .3
    #:Used for graphical radius of states
    RADIUS = .5
    # Stores the graphical pathes for states so that we can later change their
    # colors
    circles = None
    #: Number of states in the chain
    chainSize = 0
    # Y values used for drawing circles
    Y = 1
    actions_num = 2

    def __init__(self, chainSize=2):
        """
        :param chainSize: Number of states \'n\' in the chain.
        """
        self.chainSize = chainSize
        self.start = 0
        self.goal = chainSize - 1
        self.statespace_limits = np.array([[0, chainSize - 1]])
        self.episodeCap = 2 * chainSize
        super(ChainMDP, self).__init__()

    def showDomain(self, a=0):
        # Draw the environment
        s = self.state
        s = s[0]
        if self.circles is None:
            fig = plt.figure(1, (self.chainSize * 2, 2))
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
            ax.set_xlim(0, self.chainSize * 2)
            ax.set_ylim(0, 2)
            # Make the last one double circle
            ax.add_patch(
                mpatches.Circle((1 + 2 * (self.chainSize - 1), self.Y), self.RADIUS * 1.1, fc="w"))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            self.circles = [mpatches.Circle((1 + 2 * i, self.Y), self.RADIUS, fc="w")
                            for i in xrange(self.chainSize)]
            for i in xrange(self.chainSize):
                ax.add_patch(self.circles[i])
                if i != self.chainSize - 1:
                    fromAtoB(
                        1 + 2 * i + self.SHIFT,
                        self.Y + self.SHIFT,
                        1 + 2 * (i + 1) - self.SHIFT,
                        self.Y + self.SHIFT)
                    if i != self.chainSize - 2:
                        fromAtoB(
                            1 + 2 * (i + 1) - self.SHIFT,
                            self.Y - self.SHIFT,
                            1 + 2 * i + self.SHIFT,
                            self.Y - self.SHIFT,
                            'r')
                fromAtoB(
                    .75,
                    self.Y -
                    1.5 *
                    self.SHIFT,
                    .75,
                    self.Y +
                    1.5 *
                    self.SHIFT,
                    'r',
                    connectionstyle='arc3,rad=-1.2')
                plt.show()

        [p.set_facecolor('w') for p in self.circles]
        self.circles[s].set_facecolor('k')
        plt.draw()

    def step(self, a):
        s = self.state[0]
        if a == 0:  # left
            ns = max(0, s - 1)
        if a == 1:
            ns = min(self.chainSize - 1, s + 1)
        ns = np.array([ns])
        self.state = ns

        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = np.array([0])
        return self.state, self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        s = self.state
        return (s[0] == self.chainSize - 1)
