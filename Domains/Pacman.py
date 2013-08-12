from Domain import Domain
from .PacmanPackage import layout, pacman, game, ghostAgents
from .PacmanPackage import graphicsDisplay
import numpy as np
import os

######################################################
# \author Developed by Austin Hays June 18th 2013 at MIT
# The original code in PacmanPackage was taken from
#TODO
######################################################


class Pacman(Domain):
    """
    #TODO fill
    """

    _max_scared_time = 39

    actions = ["Stop", "North", "East", "South", "West"]
    num_actions = 5
    episodeCap = 1000

    def __init__(self, noise=.1, logger=None, timeout=30,
                 layoutFile='./Domains/PacmanPackage/layouts/trickyClassic.lay',
                 numGhostAgents=1000):
        """
            TODO fill
        """
        self.noise = noise
        self.logger = logger
        #Specifies which Pacman world you want
        self.layoutFile = layoutFile
        #Puts the file in line stripped format
        layout_file_content = self._tryToLoad(self.layoutFile)
        self.layout = layout.Layout(layout_file_content)
        #Number of ghosts
        self.numGhostAgents = numGhostAgents
        #Intitializes Pacman game
        self.game_state = pacman.GameState()
        self.game_rules = pacman.ClassicGameRules(timeout)
        self.game_state.data.initialize(self.layout, self.numGhostAgents)
        self.num_total_food = len(self.layout.food.asList())
        self.num_total_capsules = len(self.layout.capsules)
        self._defaultSettings()
        self.restartGraphics = None
        self.timerswitch = False
        self.savedtimer = None
        self.gameDisplay = None
        self._set_statespace_limits()
        super(Pacman, self).__init__(self.logger)

    def _set_statespace_limits(self):
        #### Makes an array of limits for each dimension in the state vector.
        statespace_limits = []
        #adds pacman x, y locations
        statespace_limits.append([1, self.layout.width - 2])
        statespace_limits.append([1, self.layout.height - 2])
        # adds ghost x, y locations and scaredTimer (how long they can be eaten)

        for ghost in self.game_state.data.agentStates[1:]:
            statespace_limits.append([1, self.layout.width - 2])
            statespace_limits.append([1, self.layout.height - 2])
            statespace_limits.append([0, self._max_scared_time])

        statespace_limits += [[0, 1]] * (self.num_total_food + self.num_total_capsules)
        self.statespace_limits = np.array(statespace_limits, dtype="float")

    def _set_state(self, s):
        """
        Takes a vector s and sets the internal game state used by the original
        pacman package.
        """

        #copies most recent state
        data = self.game_state.data
        agent_states = data.agent_states

        # set pacman position
        agent_states.configuration.pos = (s[0], s[1])

        # set ghost position
        num_ghosts = len(agent_states) - 1
        for i in range(1, num_ghosts + 1):
            part_s = s[(3*i)-1:3*i]
            agent_states[i].configuration.pos = (part_s[0], part_s[1])
            agent_states[i].scaredTimer = part_s[2]

        # set food and capsules locations
        s_food = s[(num_ghosts + 1) * 3:]
        x = 0
        y = 0
        i = 0
        data.capsules = []
        for char in str(self.layout):
            if char == ".":
                data.food[x][y] = bool(s_food[i])
                i += 1
            elif char == "o":
                coord = (x, self.layout.height - y)
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
        s = np.zeros(2 + num_ghosts * 3 + self.num_total_food + self.num_total_capsules)

        # get pacman position
        s[0:1] = agent_states[0].configuration.pos
        # get ghost info
        for i in range(num_ghosts):
            s[2 + i*3: 2 + i*3+2] = agent_states[i + 1].configuration.pos
            s[2 + i*3 + 2] = agent_states[i + 1].scaredTimer
        # get food and capsules status
        i = 2 + num_ghosts * 3
        x = 0
        y = 0
        for char in str(self.layout):
            if char == ".":
                s[i] = data.food[x][y]
                i += 1
            elif char == "\n":
                y += 1
                x = -1
            elif char == "o":
                coord = (x, self.layout.height - y)
                if coord in data.capsules:
                    s[i] = 1.
                i += 1
            x += 1
        return s
    state = property(_get_state, _set_state)

    def showDomain(self, a):
        s = self.game_state
        if self.gameDisplay is None:
            self.gameDisplay = graphicsDisplay.PacmanGraphics()
            self.gameDisplay.startGraphics(self)
            self.gameDisplay.drawStaticObjects(s.data)
            self.gameDisplay.drawAgentObjects(s.data)
        #converts s vector in pacman gamestate instance and updates
        #the display every time pacman or a ghost moves.
        #s.data.food is the correct food matrix
        s.data.layout.food = s.data.food
        for agent in range(len(s.data.agentStates)):
            s.data._agentMoved = agent
            self.gameDisplay.update(s.data)

    def step(self, a):
        """
        Applies actions from outside the Pacman domain to the given state.
        Internal states accounted for along with scoring and terminal checking.
        Returns a tuple of form (reward, state vector, terminal)
        """
        if np.random.random_sample() < self.noise:
            #Random Move
            a = np.random.choice(self.possibleActions())
        a = self.actions[a]
        #pacman performs action "a" in current state object
        pacman.PacmanRules.applyAction(self.game_state, a)
        pacman.GhostRules.checkDeath(self.game_state, 0)
        #the ghosts move randomly
        for i in range(len(self.game_state.data.agentStates))[1:]:
            ghostOptions = pacman.GhostRules.getLegalActions(self.game_state, i)
            #reverse = game.Actions.reverseDirection(self.game_state.data.agentStates[i].configuration.direction)
            #if reverse in ghostOptions and len(ghostOptions) > 1:
            #    ghostOptions.remove(reverse)
            randomAction_ind = np.random.randint(len(ghostOptions))
            randomAction = ghostOptions[randomAction_ind]
            pacman.GhostRules.applyAction(self.game_state, randomAction, i)
            pacman.GhostRules.decrementTimer(self.game_state.data.agentStates[i])
            pacman.GhostRules.checkDeath(self.game_state, i)

        #scoring in pacman
        r = self.game_state.data.scoreChange
        #r -= 1 #optional time step negative reward
        self.game_state.data.score += r
        terminal = self._is_terminal()
        if terminal:
            self.game_state.data.score = 0
        return r, self._get_state(), terminal

    def s0(self):
        """
        re-initializes internal states when an episode starts, returns a s vector
        """
        self.game_state = pacman.GameState()
        self.game_rules = pacman.ClassicGameRules(timeout=30)
        self.game = self.game_rules.newGame(self.layout, pacman, self.ghosts, DummyGraphics(), self.beQuiet, catchExceptions=False)
        self.game_state.data.initialize(self.layout, self.numGhostAgents)
        if self.gameDisplay is not None:
            self.gameDisplay.removeAllFood()
            self.gameDisplay.removeAllCapsules()
            self.gameDisplay.food = self.gameDisplay.drawFood(self.gameDisplay.layout.food)
            self.gameDisplay.capsules = self.gameDisplay.drawCapsules(self.gameDisplay.layout.capsules)

        return self.state

    def possibleActions(self, s=None):
        # beware: s is ignored

        if self._is_terminal():
            # somewhat hacky, but should not matter anyway, maybe clean up in
            # the future
            return np.array([0])
        #makes an array of possible actions pacman can perform at any given state
        possibleActions = []
        possibleMoves = pacman.GameState.getLegalActions(self.game_state, agentIndex=0)
        for a in possibleMoves:
            possibleActions.append(self.actions.index(a))
        return np.array(possibleActions)

    def isTerminal(self):
        """
        Checks whether the game should terminate at the given state.
        If game should terminate, returns the proper indication to step function.
        Accounts for scoring changes in terminal states.
        """
        return self.game_state.data._lose or self.game_state.data._win

    def _defaultSettings(self):
        self.ghostNum = 2
        self.ghosts = [ghostAgents.RandomGhost(game.Agent) for i in range(self.ghostNum)]
        self.beQuiet = False

    def _tryToLoad(self, fullname):
        #used in getLayout function
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
