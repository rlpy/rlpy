"""Gridworld Domain."""
from rlpy.Tools import plt, FONTSIZE, linearMap
import numpy as np
from .Domain import Domain
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class GridWorld(Domain):

    """
    The GridWorld domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    The map is loaded from a text file filled with numbers showing the map with the following
    coding for each cell:

    * 0: empty
    * 1: blocked
    * 2: start
    * 3: goal
    * 4: pit

    **STATE:**
    The Row and Column corresponding to the agent's location. \n
    **ACTIONS:**
    Four cardinal directions: up, down, left, right (given that
    the destination is not blocked or out of range). \n
    **TRANSITION:**
    There is 30% probability of failure for each move, in which case the action
    is replaced with a random action at each timestep. Otherwise the move succeeds
    and the agent moves in the intended direction. \n
    **REWARD:**
    The reward on each step is -.001 , except for actions
    that bring the agent to the goal with reward of +1.\n

    """

    map = start_state = goal = None
    # Used for graphics to show the domain
    agent_fig = upArrows_fig = downArrows_fig = leftArrows_fig = None
    rightArrows_fig = domain_fig = valueFunction_fig = None
    #: Number of rows and columns of the map
    ROWS = COLS = 0
    #: Reward constants
    GOAL_REWARD = +1
    PIT_REWARD = -1
    STEP_REWARD = -.001
    #: Set by the domain = min(100,rows*cols)
    episodeCap = None
    #: Movement Noise
    NOISE = 0
    # Used for graphical normalization
    MAX_RETURN = 1
    RMAX = MAX_RETURN
    # Used for graphical normalization
    MIN_RETURN = -1
    # Used for graphical shifting of arrows
    SHIFT = .1

    actions_num = 4
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6)
    #: Up, Down, Left, Right
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
    # directory of maps shipped with rlpy
    default_map_dir = os.path.join(
        __rlpy_location__,
        "Domains",
        "GridWorldMaps")

    def __init__(self, mapname=os.path.join(default_map_dir, "4x5.txt"),
                 noise=.1, episodeCap=None):
        self.map = np.loadtxt(mapname, dtype=np.uint8)
        if self.map.ndim == 1:
            self.map = self.map[np.newaxis, :]
        self.start_state = np.argwhere(self.map == self.START)[0]
        self.ROWS, self.COLS = np.shape(self.map)
        self.statespace_limits = np.array(
            [[0, self.ROWS - 1], [0, self.COLS - 1]])
        self.NOISE = noise
        self.DimNames = ['Row', 'Col']
        # 2*self.ROWS*self.COLS, small values can cause problem for some
        # planning techniques
        if not self.episodeCap:
            self.episodeCap = 1000
        else:
            self.episodeCap = episodeCap
        super(GridWorld, self).__init__()

    def showDomain(self, a=0, s=None):
        if s is None:
            s = self.state

        # Draw the environment
        if self.domain_fig is None:
            self.agent_fig = plt.figure("Domain")
            self.domain_fig = plt.imshow(
                self.map,
                cmap='GridWorld',
                interpolation='nearest',
                vmin=0,
                vmax=5)
            plt.xticks(np.arange(self.COLS), fontsize=FONTSIZE)
            plt.yticks(np.arange(self.ROWS), fontsize=FONTSIZE)
            # pl.tight_layout()
            self.agent_fig = plt.gca(
            ).plot(s[1],
                   s[0],
                   'kd',
                   markersize=20.0 - self.COLS)
            plt.show()
        self.agent_fig.pop(0).remove()
        self.agent_fig = plt.figure("Domain")
        #mapcopy = copy(self.map)
        #mapcopy[s[0],s[1]] = self.AGENT
        # self.domain_fig.set_data(mapcopy)
        # Instead of '>' you can use 'D', 'o'
        self.agent_fig = plt.gca(
        ).plot(s[1],
               s[0],
               'k>',
               markersize=20.0 - self.COLS)
        plt.draw()

    def showLearning(self, representation):
        if self.valueFunction_fig is None:
            plt.figure("Value Function")
            self.valueFunction_fig = plt.imshow(
                self.map,
                cmap='ValueFunction',
                interpolation='nearest',
                vmin=self.MIN_RETURN,
                vmax=self.MAX_RETURN)
            plt.xticks(np.arange(self.COLS), fontsize=12)
            plt.yticks(np.arange(self.ROWS), fontsize=12)
            # Create quivers for each action. 4 in total
            X = np.arange(self.ROWS) - self.SHIFT
            Y = np.arange(self.COLS)
            X, Y = np.meshgrid(X, Y)
            DX = DY = np.ones(X.shape)
            C = np.zeros(X.shape)
            C[0, 0] = 1  # Making sure C has both 0 and 1
            # length of arrow/width of bax. Less then 0.5 because each arrow is
            # offset, 0.4 looks nice but could be better/auto generated
            arrow_ratio = 0.4
            Max_Ratio_ArrowHead_to_ArrowLength = 0.25
            ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
            self.upArrows_fig = plt.quiver(
                Y,
                X,
                DY,
                DX,
                C,
                units='y',
                cmap='Actions',
                scale_units="height",
                scale=self.ROWS /
                arrow_ratio,
                width=-
                1 *
                ARROW_WIDTH)
            self.upArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS) + self.SHIFT
            Y = np.arange(self.COLS)
            X, Y = np.meshgrid(X, Y)
            self.downArrows_fig = plt.quiver(
                Y,
                X,
                DY,
                DX,
                C,
                units='y',
                cmap='Actions',
                scale_units="height",
                scale=self.ROWS /
                arrow_ratio,
                width=-
                1 *
                ARROW_WIDTH)
            self.downArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS)
            Y = np.arange(self.COLS) - self.SHIFT
            X, Y = np.meshgrid(X, Y)
            self.leftArrows_fig = plt.quiver(
                Y,
                X,
                DY,
                DX,
                C,
                units='x',
                cmap='Actions',
                scale_units="width",
                scale=self.COLS /
                arrow_ratio,
                width=ARROW_WIDTH)
            self.leftArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS)
            Y = np.arange(self.COLS) + self.SHIFT
            X, Y = np.meshgrid(X, Y)
            self.rightArrows_fig = plt.quiver(
                Y,
                X,
                DY,
                DX,
                C,
                units='x',
                cmap='Actions',
                scale_units="width",
                scale=self.COLS /
                arrow_ratio,
                width=ARROW_WIDTH)
            self.rightArrows_fig.set_clim(vmin=0, vmax=1)
            plt.show()
        plt.figure("Value Function")
        V = np.zeros((self.ROWS, self.COLS))
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones(
            (self.COLS,
             self.ROWS,
             self.actions_num),
            dtype='bool')
        arrowSize = np.zeros(
            (self.COLS,
             self.ROWS,
             self.actions_num),
            dtype='float')
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros(
            (self.COLS,
             self.ROWS,
             self.actions_num),
            dtype='uint8')
        for r in xrange(self.ROWS):
            for c in xrange(self.COLS):
                if self.map[r, c] == self.BLOCKED:
                    V[r, c] = 0
                if self.map[r, c] == self.GOAL:
                    V[r, c] = self.MAX_RETURN
                if self.map[r, c] == self.PIT:
                    V[r, c] = self.MIN_RETURN
                if self.map[r, c] == self.EMPTY or self.map[r, c] == self.START:
                    s = np.array([r, c])
                    As = self.possibleActions(s)
                    terminal = self.isTerminal(s)
                    Qs = representation.Qs(s, terminal)
                    bestA = representation.bestActions(s, terminal, As)
                    V[r, c] = max(Qs[As])
                    Mask[c, r, As] = False
                    arrowColors[c, r, bestA] = 1

                    for i in xrange(len(As)):
                        a = As[i]
                        Q = Qs[i]
                        value = linearMap(
                            Q,
                            self.MIN_RETURN,
                            self.MAX_RETURN,
                            0,
                            1)
                        arrowSize[c, r, a] = value
        # Show Value Function
        self.valueFunction_fig.set_data(V)
        # Show Policy Up Arrows
        DX = arrowSize[:, :, 0]
        DY = np.zeros((self.ROWS, self.COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 0])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 0])
        C  = np.ma.masked_array(arrowColors[:, :, 0], mask=Mask[:,:, 0])
        self.upArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Down Arrows
        DX = -arrowSize[:, :, 1]
        DY = np.zeros((self.ROWS, self.COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 1])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 1])
        C  = np.ma.masked_array(arrowColors[:, :, 1], mask=Mask[:,:, 1])
        self.downArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Left Arrows
        DX = np.zeros((self.ROWS, self.COLS))
        DY = -arrowSize[:, :, 2]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 2])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 2])
        C  = np.ma.masked_array(arrowColors[:, :, 2], mask=Mask[:,:, 2])
        self.leftArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Right Arrows
        DX = np.zeros((self.ROWS, self.COLS))
        DY = arrowSize[:, :, 3]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 3])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 3])
        C  = np.ma.masked_array(arrowColors[:, :, 3], mask=Mask[:,:, 3])
        self.rightArrows_fig.set_UVC(DY, DX, C)
        plt.draw()

    def step(self, a):
        r = self.STEP_REWARD
        ns = self.state.copy()
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

        # Take action
        ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
                ns[1] < 0 or ns[1] == self.COLS or
                self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
        else:
            # If in bounds, update the current state
            self.state = ns.copy()

        # Compute the reward
        if self.map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD

        terminal = self.isTerminal()
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.start_state.copy()
        return self.state, self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        if self.map[s[0], s[1]] == self.GOAL:
            return True
        if self.map[s[0], s[1]] == self.PIT:
            return True
        return False

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)
        for a in xrange(self.actions_num):
            ns = s + self.ACTIONS[a]
            if (
                    ns[0] < 0 or ns[0] == self.ROWS or
                    ns[1] < 0 or ns[1] == self.COLS or
                    self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        # pa: k-by-??   possible actions for each next state
        actions = self.possibleActions(s)
        k = len(actions)
        # Make Probabilities
        intended_action_index = findElemArray1D(a, actions)
        p = np.ones((k, 1)) * self.NOISE / (k * 1.)
        p[intended_action_index, 0] += 1 - self.NOISE
        # Make next states
        ns = np.tile(s, (k, 1)).astype(int)
        actions = self.ACTIONS[actions]
        ns += actions
        # Make next possible actions
        pa = np.array([self.possibleActions(sn) for sn in ns])
        # Make rewards
        r = np.ones((k, 1)) * self.STEP_REWARD
        goal = self.map[ns[:, 0], ns[:, 1]] == self.GOAL
        pit = self.map[ns[:, 0], ns[:, 1]] == self.PIT
        r[goal] = self.GOAL_REWARD
        r[pit] = self.PIT_REWARD
        # Make terminals
        t = np.zeros((k, 1), bool)
        t[goal] = True
        t[pit] = True
        return p, r, ns, t, pa

    def allStates(self):
        if self.continuous_dims == []:
            # Recall that discrete dimensions are assumed to be integer
            return (
                perms(
                    self.discrete_statespace_limits[:,
                                                    1] - self.discrete_statespace_limits[:,
                                                                                         0] + 1) + self.discrete_statespace_limits[
                    :,
                    0]
            )
