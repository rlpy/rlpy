import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.'
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Domain import Domain
from PacmanPackage import layout, pacman, game, ghostAgents, keyboardAgents
from PacmanPackage import graphicsDisplay
import numpy as np
from Tools import randSet
import os
####################################################################################
# \author Developed by Austin Hays and Christoph Dann August 2013 at MIT
# Existing Pacman code in PacmanPackage developed for CS188 at Berkeley
# by John DeNero, Dan Klein, Brad Miller, Nick Hay, and Pieter Abbeel.
# For more information visit:
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# This is a wrapper class for the original Pacman domain developed by Berkeley for edX CS-188x and introduced to us by
# Pieter Abbeel:
# https://www.edx.org/static/content-berkeley-cs188x~2013_Spring/projects/reinforcement/reinforcement.zip
####################################################################################
# State dimensions are:
# x,y for Pacman
# x,y,scared-timer for each Ghost (number of time steps for which the ghost is scared)
# binary for each location (food).   
####################################################################################




class Pacman(Domain):
    """

    """

    def __init__(self, episodeCap = None, logger = None, timeout=30,
                 prevState=None, layoutFile='./PacmanPackage/layouts/capsuleClassic.lay',
                 numGhostAgents=1000):
        self.ACTIONS                = {'Stop': 0, 'North': 1, 'East': 2, 'South': 3, 'West': 4,
                                       0: 'Stop', 1: 'North', 2: 'East', 3: 'South', 4: 'West'}
        self.pacmanLoc              = None
        self.prevGhosts             = []
        self.prevPacman             = None
        self.prevpelletLocations    = {}
        self.prevfoodLocations      = {}
        self.ghostList              = []
        self.episodeCap             = 1000
        self.logger                 = logger
        #Specifies which Pacman world you want
        self.layoutFile = layoutFile
        self.layouttext = self.tryToLoad(self.layoutFile)
        self.layout                 = layout.Layout(self.layouttext)
        self.layout_init            = layout.Layout(self.layouttext)

        self.numGhostAgents = numGhostAgents #Number of ghosts
        
        #Intitializes Pacman game
        self.state                  = pacman.GameState()
        self.rules                  = pacman.ClassicGameRules(timeout)
        self.restartGraphics        = None
        
        self.state.data.initialize(self.layout, self.numGhostAgents)
        
        #defaultSettings() initializes visualization and Pacman rules
        self.defaultSettings()
        
        self.startingIndex          = 0
        self.map                    = self.convertToInts(self.layouttext)
        self.ROWS, self.COLS        = self.map.shape
        self.statespace_limits      = self.makeStateSpace_Limits()
        self.state_space_dims       = len(self.statespace_limits)
        self.actions_num            = 5
        self.timerswitch            = False
        self.savedtimer             = None
        self.state_array            = None
        self.rlpy_array             = None
        self.gameDisplay        = None
        
        super(Pacman, self).__init__(self.logger)
        if logger:
            self.logger.log("Dims:\t\t%dx%d" % (self.ROWS, self.COLS))

    def makeStateSpace_Limits(self):
        #### Makes an array of limits for each dimension in the state vector.
        self.mapToState()
        statespace_limits = []
        #adds pacman x, y locations
        statespace_limits.append([1,self.COLS-2])
        statespace_limits.append([1,self.ROWS-2])
        # adds ghost x, y locations and scaredTimer (how long they can be eaten)
        ghostNum = len(self.state.data.agentStates[1:])
        for i in range(1, len(self.state.data.agentStates[1:])+1):
            statespace_limits.append([1,self.COLS-2])
            statespace_limits.append([1,self.ROWS-2])
            statespace_limits.append([0, 39])
        #adds binary variables for food & powerpellets
        if len(self.state.data.agentStates) > 1:
            for food in self.rlpy_array[2*len(self.state.data.agentStates)+ghostNum:]:
                statespace_limits.append([0, 1])
        else:
            for food in self.rlpy_array[2*len(self.state.data.agentStates):]:
                statespace_limits.append([0, 1])
        return np.array(statespace_limits)


    def sToObject(self,s):
        """
        Takes a vector s from step and converts it to a pacman
        self.state instance so that the internal state changes with
        RLPy state vector.
        """

        #copies most recent state
        copy = self.state.deepCopy()
        #changes pacman position as specified by s
        copy.data.agentStates[0].configuration.pos = ( s[0], s[1] )
        #saves timer for performance runs if needed
        #changes ghost positions to as specified by s
        for i in range(1, len(copy.data.agentStates[1:])+1):
            copy.data.agentStates[i].configuration.pos = ( s[(3*i)-1:][0] , s[(3*i)-1:][1])
            #copy.data.agentStates[i].configuration.pos = ( s[(3*i)-1:][0]/2 , s[(3*i)-1:][1]/2 )
            copy.data.agentStates[i].scaredTimer = s[(3*i)-1:][2]
        #updates prevfoodLocations library to values specified by s
        foodvals = s[-len(self.prevfoodLocations):]
        for key in self.prevfoodLocations.keys():
            ind = self.prevfoodLocations.keys().index(key)
            self.prevfoodLocations[key] = foodvals[ind]
        #makes an updated boolean array from updated prevfoodLocations library
        food=[]
        for i in copy.data.food:
            food.append(i)
        food.reverse()
        for x in food:
            for y in x:
                if (food.index(x), x.index(y)) not in self.prevfoodLocations:
                    pass
                #changes boolean to False if food is eaten
                elif self.prevfoodLocations[ food.index(x), x.index(y)] !=  1:
                    food[food.index(x)][x.index(y)] = False
        #updates food boolean array in pacman code
        food.reverse()
        for i in range(copy.data.food.width):
            copy.data.food[i] = food[i]
        #makes copy the state again
        self.state = copy
        #print "this is the sToObject self.state: \n", self.state
        #return self.state
        return self.state

    def mapToState(self):
        """
        Takes self.state instance and makes it into an array that
        RLPy uses.  Converts layout to numbers, then marks the positions
        of the distinguishing features of a state.
        """
        #gets current state as a layout, passes it convertToInt
        layout = self.stateTolayout(self.state.__str__())
        self.map = self.convertToInts(layout)
        #makes empty lists for location placement
        ghostLocations = []
        #pelletLocations =[]
        #powerpelletLocations = []
        state_array = []
        #finds the info to put in state array
        for i in range(self.ROWS-1):
            for j in range(self.COLS-1):
                if self.map[i,j]==2:
                    self.pacmanLoc = [j,self.ROWS-1-i]
                if self.map[i,j]==3:
                    ghostLocations.append([j,self.ROWS-1-i])
                if (self.map[i,j]==5) or (self.map[i,j]==1):
                    self.prevfoodLocations[(i,j)] = 1
        #add info to state array in correct order
        for i in self.pacmanLoc:
            state_array.append(i)
        if self.ghostList == []:
            for coord in ghostLocations:
                i=1
                for a in coord:
                    state_array.append(a)
                state_array.append(self.state.data.agentStates[i].scaredTimer)
                i+=1
        else:
            for i in self.ghostList:
                x,y,scaredTimer = i
                #x,y = int(x+0.5), int(y+0.5) #rounds to nearest integer
                state_array.append(x/2)
                state_array.append(y/2)
                state_array.append(scaredTimer)
        for loc in self.prevfoodLocations.keys():
            i,j = loc[0], loc[1]
            #if pacman eats a pellet or powerpellet
            if self.map[i,j] == 2:
                self.prevfoodLocations[(i,j)] = 0
            state_array.append(self.prevfoodLocations[(i,j)])
        #makes all array values integers, some could be floats
        self.rlpy_array = np.ceil(np.array(state_array)).astype("int")
        return self.rlpy_array

    def convertToInts(self, layout):
        #takes layout and converts it to integer form
        newarray = []
        for line in layout:
            newline = []
            for i in line:
                if i==" ":
                    newline.append(0)
                elif i==".":
                    newline.append(1)
                elif i in ["P","<",">","v","^"]:
                    newline.append(2)
                elif i=="G":
                    newline.append(3)
                elif i=="%":
                    newline.append(4)
                elif i=="o":
                    newline.append(5)
            newarray.append(newline)
        return np.array(newarray)

    def stateTolayout(self, statestring):
        #makes layout from initial state
        layout = []
        row =""
        for i in statestring:
            if i != '\n':
                row+=i
            else:
                layout.append(row)
                row=""
        #adds the last row
        layout.append(row)
        self.state_array = layout
        return self.state_array


    def showDomain(self, s, a):
        s = self.sToObject(s)
        if self.gameDisplay is None:
            self.gameDisplay = graphicsDisplay.PacmanGraphics()
            self.gameDisplay.startGraphics(self)
            self.gameDisplay.drawStaticObjects(s.data)
            self.gameDisplay.drawAgentObjects(s.data)
        #converts s vector in pacman gamestate instance and updates
        #the display every time pacman or a ghost moves.
        #s.data.food is the correct food matrix
        s.data.layout.food = s.data.food
        a = self.ACTIONS[a]
        for agent in range(len(s.data.agentStates)):
            s.data._agentMoved = agent
            self.gameDisplay.update( s.data )
            
        if self.cleanupGraphics:
            self.gameDisplay.removeAllFood()
            self.gameDisplay.removeAllCapsules()
            self.gameDisplay.food = self.gameDisplay.drawFood(self.gameDisplay.layout.food)
            self.gameDisplay.capsules = self.gameDisplay.drawCapsules(self.gameDisplay.layout.capsules)
            self.cleanupGraphics = False
        
    def step(self, s, a):
        """
        Applies actions from outside the Pacman domain to the given state.
        Internal states accounted for along with scoring and terminal checking.
        Returns a tuple of form (reward, state vector, terminal)
        """
        #matches internal states with given s vector
        self.state = self.sToObject(s)
        a = self.ACTIONS[a]
        #pacman performs action "a" in current state object
        pacman.PacmanRules.applyAction( self.state, a )
        s = self.mapToState()
        #the ghosts move according to PacmanPackage
        for i in range(len(self.state.data.agentStates))[1:]:
            conf = self.state.getGhostState( i ).configuration
            if len(self.ghostList) == len(self.state.data.agentStates)-1:
                conf.pos = self.ghostList[i-1][0]/2.0, self.ghostList[i-1][1]/2.0
            ghostOptions = pacman.GhostRules.getLegalActions(self.state ,i)
            randomAction_ind = np.random.randint(len(ghostOptions))
            randomAction = ghostOptions[randomAction_ind]
            pacman.GhostRules.applyAction(self.state,randomAction,i)
            pacman.GhostRules.decrementTimer(self.state.data.agentStates[i])
            pacman.GhostRules.checkDeath(self.state, i)
            # KEEP ghostList THE SAME, it ensures the ghosts' speed is correct
            if len(self.ghostList) < i:
                self.ghostList.append((self.state.data.agentStates[i].configuration.pos[0]*2,self.state.data.agentStates[i].configuration.pos[1]*2, self.state.data.agentStates[i].scaredTimer))
            self.ghostList[i-1] = (self.state.data.agentStates[i].configuration.pos[0]*2,self.state.data.agentStates[i].configuration.pos[1]*2, self.state.data.agentStates[i].scaredTimer)
        s = self.mapToState()
        #scoring in pacman
        r = self.state.data.scoreChange
        if len(self.state.data.agentStates) > 1:
            if self.state.data.agentStates[1].scaredTimer == 39:
                r+=50
        r -= 1 #optional time step negative reward
        self.state.data.score += r
        terminal = self.isTerminal(s)
        
        #self.mapToState updates state_array for RLPy use
        return r, self.rlpy_array, terminal
    
        

    def s0(self):
        #re-initializes internal states when an episode starts, returns a s vector
        self.ghostDict = {}
        self.state.data.score = 0
        self.cleanupGraphics = True
        self.ghostDict = {}
        self.ghostList = []
        self.state = pacman.GameState()
        self.rules = pacman.ClassicGameRules(timeout=30)
        self.game = self.rules.newGame(self.layout, pacman, self.ghosts, DummyGraphics(), self.beQuiet, catchExceptions=False)
        self.state.data.initialize(self.layout, self.numGhostAgents)

#        if self.gameDisplay is not None:
#            self.gameDisplay.removeAllFood()
#            self.gameDisplay.removeAllCapsules()
#            self.gameDisplay.food = self.gameDisplay.drawFood(self.gameDisplay.layout.food)
#            self.gameDisplay.capsules = self.gameDisplay.drawCapsules(self.gameDisplay.layout.capsules)

        return self.mapToState()

    def possibleActions(self, s):
        if self.isTerminal(s):
            # somewhat hacky, but should not matter anyway, maybe clean up in
            # the future
            return np.array([0])
        #makes an array of possible actions pacman can perform at any given state
        possibleActions=[]
        possibleMoves = pacman.GameState.getLegalActions(self.state, agentIndex=0)
        for a in possibleMoves:
            possibleActions.append(self.ACTIONS[a])
        return np.array(possibleActions)

    def isTerminal(self, state):
        """
        Checks whether the game should terminate at the given state.
        If game should terminate, returns the proper indication to step function.
        Accounts for scoring changes in terminal states.
        """
#        #pacman and a ghost are in the same place --> pacman dies, game is over
#        pacman = self.state.data.agentStates[0].getPosition()
#        for ghost in self.state.data.agentStates[1:]:
#            #checks to see if they crossed each other in previous step
#            if ghost.getPosition() == self.prevPacman:
#                if pacman in self.prevGhosts:
#                    if ghost.scaredTimer == 0:
#                        self.state.data.scoreChange -= 500
#                        state = self.s0()
#                        return True
#            self.prevGhosts = []
#            self.prevGhosts.append(ghost.getPosition())
#            #checks if pacman and ghost are in the same place
#            if ghost.getPosition() == pacman:
#                if ghost.scaredTimer == 0:
#                    self.state.data.scoreChange -= 500
#                    return True
#        #checks if pacman has eaten everything, checks that every food indicator variable is zero
#        if 1 not in self.rlpy_array[2*len(self.state.data.agentStates):]:
#            self.state.data.scoreChange += 500
#            return True
        #checks internal winning mechanism if something escapes above tests
        if self.state.data._win is True:
            self.state.data.scoreChange += 500
            return True
        if self.state.data._lose is True:
            self.state.data.scoreChange -= 500
            return True
        self.prevPacman = pacman
        return False

    def defaultSettings(self):
        self.ghostNum = 2
        self.ghosts = [ghostAgents.RandomGhost(game.Agent) for i in range(self.ghostNum)]

        self.beQuiet = False
        self.startingIndex = 0

    def getLayout(self, name, back = 2):
        #loads the layout from the given file ending in .lay
        if name.endswith('.lay'):
            layout = self.tryToLoad('Domains/PacmanPackage/layouts/' + name)
            if layout is None:
                layout = self.tryToLoad('Domains/PacmanPackage/layouts/' + name)
        else:
            layout = self.tryToLoad('Domains/PacmanPackage/layouts/' + name + '.lay')
            if layout is None:
                layout = self.tryToLoad(name + '.lay')
        if layout is None and back >= 0:
            curdir = os.path.abspath('.')
            os.chdir('..')
            layout = self.getLayout(name, back -1)
            os.chdir(curdir)
        return layout

    def tryToLoad(self, fullname):
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


if __name__ == "__main__":
    #runs interactive game with whatever layout is initialized in the Domain subclass
    p = Pacman()
    p.test(1000)
    
    os.chdir(RL_PYTHON_ROOT+'/Domains/PacmanPackage/')
    args = {}
    args['layout'] = layout.getLayout(p.layoutFile.split('/')[-1])
    args['ghosts'] = [ghostAgents.RandomGhost(i) for i in range(len(p.state.data.agentStates))[1:]]
    args['numGames'] = 1
    args['pacman'] = keyboardAgents.KeyboardAgent(0)
    args['catchExceptions'] = False
    args['record'] = False
    args['timeout'] = 30
    args['display'] = graphicsDisplay.PacmanGraphics(1.0, 0.1)
    pacman.runGames( **args )
