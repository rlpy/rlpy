"""Pacman game domain."""
from rlpy.Tools import __rlpy_location__
from .Domain import Domain
from .PacmanPackage import layout, pacman, game, ghostAgents
from .PacmanPackage import graphicsDisplay
import numpy as np
from copy import deepcopy
import os
import time

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Austin Hays"


class Pacman(Domain):

    """
    Pacman domain, which acts as a wrapper for the Pacman implementation
    from the BerkeleyX/CS188.1x course project 3.

    **STATE:** The state vector has a series of dimensions:

    * [2] The x and y coordinates of pacman
    * [3 * ng] the x and y coordinates as well as the scare time of each ghost
      ("scare time" is how long the ghost remains scared after consuming a capsule.)
    * [nf] binary variables indicating if a food is still on the board or not
    * [nc] binary variables for each capsule indicating if it is still on the board or not

    *nf* and *nc* are map-dependent, and *ng* can be set as a parameter.
    Based on above, total dimensionality of state vector is map-dependent,
    and given by (2 + 3*ng + nf + nc).

    **ACTIONS:** Move Pacman [up, down, left, right, stay]

    **REWARD:** See the Berkeley project website below for more info.

    .. note::
        The visualization runs as fast as your CPU will permit; to slow things
        down so gameplay is actually visible, de-comment time.sleep()
        in the showDomain() method.

    **REFERENCE:** This domain is an RLPy wrapper for the implementation
    from the `BerkeleyX/CS188.1x course project 3 <https://courses.edx.org/courses/BerkeleyX/CS188.1x/2013_Spring/courseware/Week_9/Project_3_Reinforcement/>`_

    See the original `source code (zipped) <https://courses.edx.org/static/content-berkeley-cs188x~2013_Spring/projects/reinforcement/reinforcement.zip>`_

    For more details of the domain see the original package in the `Domains/PacmanPackage` folder.

    """

    _max_scared_time = 39

    actions = ["Stop", "North", "East", "South", "West"]
    actions_num = 5
    episodeCap = 1000

    #: location of layouts shipped with rlpy
    default_layout_dir = os.path.join(
        __rlpy_location__, "Domains", "PacmanPackage",
        "layouts")

    def __init__(self, noise=.1, timeout=30,
                 layoutFile=os.path.join(
                     default_layout_dir, 'trickyClassic.lay'),
                 numGhostAgents=1000):
        """
        layoutFile:
            filename of the map file
        noise:
            with this probability pacman makes a random move instead the one
            specified by the action
        """
        self.noise = noise
        # Specifies which Pacman world you want
        self.layoutFile = layoutFile
        # Puts the file in line stripped format
        layout_file_content = self._tryToLoad(self.layoutFile)
        self.layout = layout.Layout(layout_file_content)
        # Number of ghosts
        self.numGhostAgents = numGhostAgents
        # Intitializes Pacman game
        self.game_state = pacman.GameState()
        self.game_rules = pacman.ClassicGameRules(timeout)
        self.layout_copy = deepcopy(self.layout)
        self.game_state.data.initialize(self.layout_copy, self.numGhostAgents)
        self.num_total_food = len(self.layout_copy.food.asList())
        self.num_total_capsules = len(self.layout_copy.capsules)
        self._defaultSettings()
        self.restartGraphics = None
        self.timerswitch = False
        self.savedtimer = None
        self.gameDisplay = None
        self._set_statespace_limits()
        super(Pacman, self).__init__()

    def _set_statespace_limits(self):
        # Makes an array of limits for each dimension in the state vector.
        statespace_limits = []
        # adds pacman x, y locations
        statespace_limits.append([1, self.layout.width - 2])
        statespace_limits.append([1, self.layout.height - 2])
        # adds ghost x, y locations and scaredTimer (how long they can be
        # eaten)

        for ghost in self.game_state.data.agentStates[1:]:
            statespace_limits.append([1, self.layout.width - 2])
            statespace_limits.append([1, self.layout.height - 2])
            statespace_limits.append([0, self._max_scared_time])

        statespace_limits += [[0, 1]] * (
            self.num_total_food + self.num_total_capsules)
        self.statespace_limits = np.array(statespace_limits, dtype="float")

    def _set_state(self, s):
        """
        Takes a vector s and sets the internal game state used by the original
        pacman package.
        """

        # copies most recent state
        data = self.game_state.data
        agent_states = data.agentStates

        # set pacman position
        agent_states.configuration.pos = (s[0], s[1])

        # set ghost position
        num_ghosts = len(agent_states) - 1
        for i in range(1, num_ghosts + 1):
            part_s = s[(3 * i) - 1:3 * i]
            agent_states[i].configuration.pos = (part_s[0], part_s[1])
            agent_states[i].scaredTimer = part_s[2]

        # set food and capsules locations
        s_food = s[(num_ghosts + 1) * 3:]
        x = 0
        y = 0
        i = 0
        data.capsules = []
        for char in str(self.layout_copy):
            if char == ".":
                data.food[x][y] = bool(s_food[i])
                i += 1
            elif char == "o":
                coord = (x, self.layout_copy.height - y)
                if s_food[i]:
                    data.capsules.append(coord)
                i += 1
            elif char == "\n":
                y += 1
                x = -1
            x += 1

    def _get_state(self):
        """
        get the internal game state represented as a numpy array
        """
        data = self.game_state.data
        agent_states = self.game_state.data.agentStates
        num_ghosts = len(agent_states) - 1
        s = np.zeros(
            2 + num_ghosts * 3 + self.num_total_food + self.num_total_capsules)

        # get pacman position
        s[:2] = agent_states[0].configuration.pos
        # import ipdb; ipdb.set_trace()
        # get ghost info
        for i in range(num_ghosts):
            s[2 + i * 3: 2 + i * 3 + 2] = agent_states[i + 1].configuration.pos
            s[2 + i * 3 + 2] = agent_states[i + 1].scaredTimer
        # get food and capsules status
        i = 2 + num_ghosts * 3
        x = 0
        y = 0
        for char in str(self.layout_copy):
            if char == ".":
                s[i] = data.food[x][y]
                i += 1
            elif char == "\n":
                y += 1
                x = -1
            elif char == "o":
                coord = (x, self.layout_copy.height - y)
                if coord in data.capsules:
                    s[i] = 1.
                i += 1
            x += 1
        return s
    state = property(_get_state, _set_state)

    def showDomain(self, a, s=None):
        if s is not None:
            errStr = 'ERROR: In Pacman.py, attempted to pass a state (s)'\
                'to showDomain(); Pacman only supports internal states.'\
                'If you do pass a state parameter, ensure it is set to None.'
            raise Exception(errStr)
        s = self.game_state
        if self.gameDisplay is None:
            self.gameDisplay = graphicsDisplay.PacmanGraphics()
            self.gameDisplay.startGraphics(self)
            self.gameDisplay.drawStaticObjects(s.data)
            self.gameDisplay.drawAgentObjects(s.data)
        elif self._cleanup_graphics:
            self._cleanup_graphics = False
            self.gameDisplay.removeAllFood()
            self.gameDisplay.removeAllCapsules()
            self.gameDisplay.food = self.gameDisplay.drawFood(
                self.gameDisplay.layout.food)
            self.gameDisplay.capsules = self.gameDisplay.drawCapsules(
                self.gameDisplay.layout.capsules)
        # converts s vector in pacman gamestate instance and updates
        # the display every time pacman or a ghost moves.
        # s.data.food is the correct food matrix
        s.data.layout.food = s.data.food
        for agent in range(len(s.data.agentStates)):
            s.data._agentMoved = agent
            self.gameDisplay.update(s.data)
            s._foodEaten = None
            s._capsuleEaten = None
# time.sleep(0.1) # Sleep for 0.1 sec

    def step(self, a):
        """
        Applies actions from outside the Pacman domain to the given state.
        Internal states accounted for along with scoring and terminal checking.
        Returns a tuple of form (reward, new state vector, terminal)
        """
        if self.random_state.random_sample() < self.noise:
            # Random Move
            a = self.random_state.choice(self.possibleActions())
        a = self.actions[a]
        next_state_p = self.game_state.generateSuccessor(0, a)
        next_state = next_state_p
        # pacman performs action "a" in current state object
        # pacman.PacmanRules.applyAction(self.game_state, a)
        # pacman.GhostRules.checkDeath(self.game_state, 0)
        # the ghosts move randomly
        for i in range(1, len(self.game_state.data.agentStates)):
            if next_state.isWin() or next_state.isLose():
                break
            ghostOptions = pacman.GhostRules.getLegalActions(next_state, i)
            # TODO: use domain random stream
            randomAction_ind = self.random_state.randint(len(ghostOptions))
            randomAction = ghostOptions[randomAction_ind]
            next_state = next_state.generateSuccessor(i, randomAction)
        # keep track of eaten stuff for graphics (original code assumes
        # graphics are updated after every agent's move)
        next_state.data._foodEaten = next_state_p.data._foodEaten
        next_state.data._capsuleEaten = next_state_p.data._capsuleEaten
        # scoring in pacman
        r = next_state.data.score - self.game_state.data.score
        self.game_state = next_state
        terminal = self.isTerminal()
        return r, self._get_state(), terminal, self.possibleActions()

    def s0(self):
        """
        re-initializes internal states when an episode starts, returns a s vector
        """
        self.game_state = pacman.GameState()
        self.game_rules = pacman.ClassicGameRules(timeout=30)
        self.layout_copy = deepcopy(self.layout)
        self.game = self.game_rules.newGame(
            self.layout_copy, pacman, self.ghosts, DummyGraphics(), self.beQuiet, catchExceptions=False)
        self.game_state.data.initialize(self.layout_copy, self.numGhostAgents)
        self._cleanup_graphics = True

        return self.state, self.isTerminal(), self.possibleActions()

    def possibleActions(self):

        if self.isTerminal():
            # somewhat hacky, but should not matter anyway, maybe clean up in
            # the future
            return np.array([0])
        # makes an array of possible actions pacman can perform at any given
        # state
        possibleActions = []
        possibleMoves = pacman.GameState.getLegalActions(
            self.game_state, agentIndex=0)
        for a in possibleMoves:
            possibleActions.append(self.actions.index(a))
        return np.array(possibleActions)

    def isTerminal(self):
        """
        Checks whether the game should terminate at the given state.
        (Terminate for failure, ie eaten by ghost or out of time, and for
        success, all food on map eaten.)
        If game should terminate, returns the proper indication to step function.
        Accounts for scoring changes in terminal states.
        """
        return self.game_state.data._lose or self.game_state.data._win

    def _defaultSettings(self):
        self.ghostNum = 2
        self.ghosts = [ghostAgents.RandomGhost(
            game.Agent) for i in range(self.ghostNum)]
        self.beQuiet = False

    def _tryToLoad(self, fullname):
        # used in getLayout function
        f = open(fullname)
        grid = [line.strip() for line in f]
        f.close()
        return grid


class DummyGraphics(object):

    def initialize(self, *arg, **kwargs):
        pass

    def update(self, *arg, **kwargs):
        pass

    def finalize(self, *arg, **kwargs):
        pass
