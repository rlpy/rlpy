"""BlocksWorld domain, stacking of blocks to form a tower."""

from .Domain import Domain
from rlpy.Tools import id2vec, vec2id, findElemArray1D
from rlpy.Tools import nchoosek, factorial, findElemArray2D, plt, FONTSIZE
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class BlocksWorld(Domain):

    """
    Classical BlocksWorld Domain [Winograd, 1971].

    The objective is to put blocks on top of each other in a specific order to form
    a tower. Initially all blocks are unstacked and are on the table.
    **STATE:**
    The state of the MDP is defined by n integer values [s_1 ... s_n]: si = j indicates
    that block i is on top of j (for compactness s_i = i indicates that the block i
    is on the table).

    [0 1 2 3 4 0] => means all blocks on table except block 5 which is on top of block 0

    **ACTIONS:**
    At each step, the agent can take a block, and put it on top of another
    block or move it to the table, given that blocks do not have any other blocks
    on top of them prior to this action.

    **TRANSITION:**
    There is 30% probability of failure for each move, in which case the agent drops
    the moving block on the table. Otherwise the move succeeds.

    **REWARD:**
    The reward is -.001 for each step where the tower is not built and +1.0
    when the tower is built.

    **REFERENCE:**

    .. seealso::
        Alborz Geramifard, Finale Doshi, Joshua Redding, Nicholas Roy, and Jonathan How.
        Online discovery of feature dependencies.
        International Conference on Machine Learning (ICML), pages 881-888.
        ACM, June 2011

    """

    #: reward per step
    STEP_REWARD = -.001
    #: reward when the tower is completed
    GOAL_REWARD = 1
    #: discount factor
    discount_factor = 1
    #: Total number of blocks
    blocks = 0
    #: Goal tower size
    towerSize = 0
    episodeCap = 1000
    #: Used to plot the domain
    domain_fig = None

    def __init__(self, blocks=6, towerSize=6, noise=.3):
        self.blocks = blocks
        self.towerSize = towerSize
        self.noise = noise
        self.TABLE = blocks + 1
        self.actions_num = blocks * blocks
        self.discount_factor = 1
        # Block i is on top of what? if block i is on top of block i => block i
        # is on top of table
        self.statespace_limits = np.tile([0, blocks - 1], (blocks, 1))
        # This is the true size of the state space refer to [Geramifard11_ICML]
        self.real_states_num = sum(
            [nchoosek(blocks,
                      i) * factorial(blocks - i) * pow(i,
                                                       blocks - i) for i in xrange(blocks)])
        # [0 0 1 2 3 .. blocks-2] meaning block 0 on the table and all other stacked on top of e
        self.GOAL_STATE = np.hstack(([0], np.arange(0, blocks - 1)))
        # Make Dimension Names
        self.DimNames = []
        for a in xrange(blocks):
            self.DimNames.append(['%d on' % a])
        super(BlocksWorld, self).__init__()

    def showDomain(self, a=0):
        # Draw the environment
        s = self.state
        world = np.zeros((self.blocks, self.blocks), 'uint8')
        undrawn_blocks = np.arange(self.blocks)
        while len(undrawn_blocks):
            A = undrawn_blocks[0]
            B = s[A]
            undrawn_blocks = undrawn_blocks[1:]
            if B == A:  # => A is on Table
                world[0, A] = A + 1  # 0 is white thats why!
            else:
                # See if B is already drawn
                i, j = findElemArray2D(B + 1, world)
                if len(i):
                    world[i + 1, j] = A + 1  # 0 is white thats why!
                else:
                    # Put it in the back of the list
                    undrawn_blocks = np.hstack((undrawn_blocks, [A]))
        if self.domain_fig is None:
            self.domain_fig = plt.imshow(
                world,
                cmap='BlocksWorld',
                origin='lower',
                interpolation='nearest')  # ,vmin=0,vmax=self.blocks)
            plt.xticks(np.arange(self.blocks), fontsize=FONTSIZE)
            plt.yticks(np.arange(self.blocks), fontsize=FONTSIZE)
            # pl.tight_layout()
            plt.axis('off')
            plt.show()
        else:
            self.domain_fig.set_data(world)
            plt.draw()

    def showLearning(self, representation):
        pass  # cant show 6 dimensional value function

    def step(self, a):
        s = self.state
        # move block A on top of B
        [A, B] = id2vec(a, [self.blocks, self.blocks])
        # print 'taking action %d->%d' % (A,B)
        if not self.validAction(s, A, B):
            print 'State:%s, Invalid move from %d to %d' % (str(s), A, B)
            print self.possibleActions()
            print id2vec(self.possibleActions(), [self.blocks, self.blocks])

        if self.random_state.random_sample() < self.noise:
            B = A  # Drop on Table
        ns = s.copy()
        ns[A] = B  # A is on top of B now.
        self.state = ns.copy()
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        # all blocks on table
        self.state = np.arange(self.blocks)
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def possibleActions(self):
        s = self.state
        # return the id of possible actions
        # find empty blocks (nothing on top)
        empty_blocks = [b for b in xrange(self.blocks) if self.clear(b, s)]
        actions = [[a,
                    b] for a in empty_blocks for b in empty_blocks if not self.destination_is_table(
            a,
            b) or not self.on_table(a,
                                    s)]  # condition means if A sits on the table you can not pick it and put it on the table
        return np.array([vec2id(x, [self.blocks, self.blocks]) for x in actions])

    def validAction(self, s, A, B):
        # Returns true if B and A are both empty.
        return (
            (self.clear(A, s)
             and (self.destination_is_table(A, B) or self.clear(B, s)))
        )

    def isTerminal(self):
        return np.array_equal(self.state, self.GOAL_STATE)

    def top(self, A, s):
        # returns the block on top of block A. Return [] if nothing is on top
        # of A
        on_A = findElemArray1D(A, s)
        on_A = np.setdiff1d(on_A, [A])  # S[i] = i is the key for i is on table.
        return on_A

    def clear(self, A, s):
        # returns true if block A is clear and can be moved
        return len(self.top(A, s)) == 0

    def destination_is_table(self, A, B):
        # See for move A->B, B is table
        return (A == B)

    def on_table(self, A, s):
        # returns true of A is on the table
        return s[A] == A

    def towerTop(self, A, s):
        # inspect block A and return the highest block which is stacked over A.
        # Hence if B is on A, and C is on B, this function returns C
        # If A is clear => returns A itself
        block = A
        while True:
            highestTop = self.top(block, s)
            if len(highestTop) == 0:
                break
            else:
                block = highestTop[0]
        return block

    def on(self, A, B, s):
        # returns true if block A is on block B
        return s[A] == B

    def getActionPutAonTable(self, A):
        return vec2id(np.array([A, A]), [self.blocks, self.blocks])

    def getActionPutAonB(self, A, B):
        return vec2id(np.array([A, B]), [self.blocks, self.blocks])

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        [A, B] = id2vec(a, [self.blocks, self.blocks])
        # Nominal Move:
        ns1 = s.copy()
        ns1[A] = B  # A is on top of B now.
        terminal1 = self.isTerminal(ns1)
        r1 = self.GOAL_REWARD if terminal1 else self.STEP_REWARD
        if self.destination_is_table(A, B):
            p = np.array([1]).reshape((1, -1))
            r = np.array([r1]).reshape((1, -1))
            ns = np.array([ns1]).reshape((1, -1))
            t = np.array([terminal1]).reshape((1, -1))
            return p, r, ns, t
        else:
            # consider dropping the block
            ns2 = s.copy()
            ns2[A] = A  # Drop on table
            terminal2 = self.isTerminal(ns2)
            r2 = self.GOAL_REWARD if terminal2 else self.STEP_REWARD
            p = np.array([1 - self.noise, self.noise]).reshape((2, 1))
            r = np.array([r1, r2]).reshape((2, 1))
            ns = np.array([[ns1], [ns2]]).reshape((2, -1))
            t = np.array([terminal1, terminal2]).reshape((2, -1))
            return p, r, ns, t
