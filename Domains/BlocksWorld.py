"""BlocksWorld domain, stacking of blocks to form a tower"""
from Domains.Domain import Domain
import Tools
from collections import defaultdict
from copy import copy
import numpy as np
from Tools import memory, plt
from Tools.progressbar import ProgressBar
__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
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
    is on the table). \n
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
    gamma = 1
    #: Total number of blocks
    blocks = 0
    #: Goal tower size
    towerSize = 0
    episodeCap = 1000
    #: Used to plot the domain
    domain_fig = None

    def __init__(self, blocks=6, towerSize=6, noise=.3, logger=None, gamma=1.):
        self.blocks = blocks
        self.towerSize = towerSize
        self.noise = noise
        self.TABLE = blocks + 1
        self.actions_num = blocks * blocks
        self.gamma = gamma
        self.statespace_limits = Tools.tile([0, blocks - 1], (blocks, 1))
        # Block i is on top of what? if block i is on top of block i => block i is on top of table
        self.real_states_num = sum([Tools.nchoosek(blocks, i) * Tools.factorial(blocks - i) * pow(i, blocks - i)
                                    for i in np.arange(blocks)])  # This is the true size of the state space refer to [Geramifard11_ICML]
        self.GOAL_STATE = np.hstack(([0], np.arange(0, blocks - 1)))
        # [0 0 1 2 3 .. blocks-2] meaning block 0 on the table and all other stacked on top of e
        # Make Dimension Names
        self.DimNames = []
        for a in np.arange(blocks):
            self.DimNames.append(['%d on' % a])
        super(BlocksWorld, self).__init__(logger)
        if logger:
            self.logger.log("noise\t\t%0.1f" % self.noise)
            self.logger.log("blocks\t\t%d" % self.blocks)

        def gk(key):
            key["policy"] = (key["policy"].__class__.__name__,
                            copy(key["policy"].__getstate__()))
            if "domain" in key["policy"][1]:
                del key["policy"][1]["domain"]

            key["self"] = copy(key["self"].__getstate__())
            lst = [
                "random_state", "V", "domain_fig", "logger", "DirNames", "state",
                "transition_samples", "_all_states", "chain_model", "policy"]
            for l in lst:
                if l in key["self"]:
                    del key["self"][l]
            return key

        def ggk(key):
            key["self"] = key["self"].blocks
            return key
        self.V = memory.cache(self.V, get_key=gk)
        self.chain_model = memory.cache(self.chain_model, get_key=gk)
        self._all_states = memory.cache(self._all_states, get_key=ggk)
        self.transition_samples = memory.cache(self.transition_samples, get_key=gk)
        self.s0()

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
                world[0, A] = A+1  # 0 is white thats why!
            else:
                # See if B is already drawn
                i, j = Tools.findElemArray2D(B+1, world)
                if len(i):
                    world[i+1, j] = A+1  # 0 is white thats why!
                else:
                    # Put it in the back of the list
                    undrawn_blocks = np.hstack((undrawn_blocks, [A]))
        if self.domain_fig is None:
            self.domain_fig = plt.imshow(
                world, cmap='BlocksWorld', origin='lower', interpolation='nearest')  # ,vmin=0,vmax=self.blocks)
            plt.xticks(np.arange(self.blocks), fontsize=Tools.FONTSIZE)
            plt.yticks(np.arange(self.blocks), fontsize=Tools.FONTSIZE)
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
        """
        #FIXME HACK, TEMPORARY FOR DEBUGGING
        P, R, states, terminal = self.chain_model(self.policy)
        i = self._find_state_idx(s, states)
        j = self.random_state.choice(np.arange(P.shape[1]), p = P[i])
        ns = states[j]

        """
        [A, B] = Tools.id2vec(a, [
                        self.blocks, self.blocks])  # move block A on top of B
        # print 'taking action %d->%d' % (A,B)
        if not self.validAction(s, A, B):
            print 'State:%s, Invalid move from %d to %d' % (str(s), A, B)
            print self.possibleActions()
            print Tools.id2vec(self.possibleActions(), [self.blocks, self.blocks])

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

    @staticmethod
    def state_valid(s):
        """checks if a given array is a valid state
        Does not check length of the array
        simple dfs to check for loops in the on-top-of graph"""
        n = len(s)
        pred = np.ones(n, dtype="int") * (-1)
        for i in xrange(n):
            j = i
            while s[j] != j:
                k = s[j]
                if k < 0 or k >= n:
                    # invalid value
                    return False
                if pred[k] != -1 and pred[k] != j:
                    # two blocks on one
                    return False
                if k == i:
                    # loop
                    return False
                pred[k] = j
                j = k
        return True

    def _all_states(self):
        # wont deal with more than 255 blocks
        assert self.blocks < 255
        max_num_s = Tools.factorial(self.blocks) * (self.blocks + 1)
        states = np.zeros((max_num_s, self.blocks), dtype="uint8")
        s = np.ones(self.blocks, dtype="uint8") * -1
        k = 0
        d = 0
        while d >= 0:
            if d == self.blocks - 1 and s[d] != -1:
                if self.state_valid(s):
                    states[k] = s
                    k += 1
            if s[d] < self.blocks - 1:
                s[d] += 1
                if d < self.blocks - 1:
                    d += 1
            else:
                s[d] = -1
                d -= 1
        return states[:k]

    @staticmethod
    def _find_state_idx(s, states):
        """finds the row index of s in states
        assumes that the states are ordered"""
        for i in xrange(states.shape[0]):
            if np.all(states[i] == s):
                return i
        return -1

    def stationary_distribution(self, policy, states, num_traj=10000):
        stat = self.state.copy()
        ran_stat = copy(self.random_state)
        counts = defaultdict(int)
        weights = np.power(10,np.arange(self.blocks-1, -1, -1))
        with ProgressBar() as p:
            for i in xrange(num_traj):
                self.s0()
                s, _, _ = self.sample_trajectory(
                    self.state, policy, self.episodeCap)
                k = (s * weights).sum(axis=1)
                for j in xrange(len(k)):
                    counts[k[j]] += 1
                p.update(i, num_traj, "Sample for Stat. Distr.")
        ordered_counts = np.zeros(states.shape[0])
        k = (states * weights).sum(axis=1)
        for i in xrange(len(k)):
            ordered_counts[i] = counts[k[i]]

        self.state = stat
        self.random_state = ran_stat

        return ordered_counts

    def chain_model_from_samples(self, policy, num_samples=10000):
        states = self._all_states()
        n = states.shape[0]
        P = np.zeros((n, n))
        R = np.zeros(n)
        counts = np.zeros(n)
        terminal = np.zeros(n, dtype="bool")
        for i in xrange(n):
            if self.isTerminal(states[i]):
                terminal[i] = True
                P[i, i] = 1.

        S, S_term, Sn, Sn_term, Rs = self.transition_samples(policy, num_samples=num_samples)
        for i in xrange(len(S_term)):
            s = S[i]
            sn = Sn[i]
            k = self._find_state_idx(s, states)
            l = self._find_state_idx(sn, states)
            counts[k] += 1
            P[k, l] += 1
            R[k] += Rs[i]
        R[counts > 0] /= counts[counts > 0]
        P[counts > 0] /= counts[counts > 0][:,None]
        return P, R, states, terminal

    def chain_model(self, policy):
        states = self._all_states()
        n = states.shape[0]
        P = np.zeros((n, n))
        R = np.zeros(n)
        terminal = np.zeros(n, dtype="bool")
        with ProgressBar() as p:
            for i in xrange(n):
                s = states[i]
                p_actions = self.possibleActions(s)
                term = self.isTerminal(s)
                terminal[i] = term
                if term:
                    P[i, i] = 1.
                    continue
                pi = policy.prob(s, term, p_actions)
                assert np.allclose(pi.sum(), 1.)
                for a in p_actions:
                    [A, B] = Tools.id2vec(a, (self.blocks, self.blocks))
                    ns = s.copy()
                    ns[A] = B
                    j = self._find_state_idx(ns, states)
                    assert j >= 0
                    r = self.GOAL_REWARD if self.isTerminal(
                        ns) else self.STEP_REWARD
                    R[i] += pi[a] * r * (1. - self.noise)
                    P[i, j] += (1. - self.noise) * pi[a]
                    ns[A] = A
                    j = self._find_state_idx(ns, states)
                    assert j >= 0
                    r = self.GOAL_REWARD if self.isTerminal(
                        ns) else self.STEP_REWARD
                    R[i] += pi[a] * r * self.noise
                    P[i, j] += self.noise * pi[a]
                p.update(i, n, "Build Model")
        return P, R, states, terminal

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        # return the id of possible actions
        # find empty blocks (nothing on top)
        empty_blocks = [b for b in np.arange(self.blocks) if self.clear(b, s)]
        actions = [[a, b] for a in empty_blocks for b in empty_blocks if not self.destination_is_table(
            a, b) or not self.on_table(a, s)]  # condition means if A sits on the table you can not pick it and put it on the table
        return np.array([Tools.vec2id(x, [self.blocks, self.blocks]) for x in actions])

    def V(self, policy, discretization=None, max_len_traj=None, num_traj=None):
        P, R, states, s_term = self.chain_model(policy)
        V = np.linalg.solve(np.eye(P.shape[0]) - self.gamma * P, R)
        s_distr = self.stationary_distribution(policy, states)
        return V, states.T, s_term, s_distr

    def validAction(self, s, A, B):
        # Returns true if B and A are both empty.
        return (self.clear(A, s) and (self.destination_is_table(A, B) or self.clear(B, s)))

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return np.array_equal(s, self.GOAL_STATE)

    def top(self, A, s):
        # returns the block on top of block A. Return [] if nothing is on top
        # of A
        on_A = Tools.findElemArray1D(A, s)
        on_A = Tools.setdiff1d(on_A, [A])  # S[i] = i is the key for i is on table.
        return on_A

    def clear(self, A, s):
        # returns true if block A is clear and can be moved
        return len(self.top(A, s)) == 0

    def destination_is_table(self, A, B):
        # See for move A->B, B is table
        return A == B

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
        return Tools.vec2id(np.array([A, A]), [self.blocks, self.blocks])

    def getActionPutAonB(self, A, B):
        return Tools.vec2id(np.array([A, B]), [self.blocks, self.blocks])

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        [A, B] = Tools.id2vec(a, [self.blocks, self.blocks])
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
            p = np.array([1-self.noise, self.noise]).reshape((2, 1))
            r = np.array([r1, r2]).reshape((2, 1))
            ns = np.array([[ns1], [ns2]]).reshape((2, -1))
            t = np.array([terminal1, terminal2]).reshape((2, -1))
            return p, r, ns, t
