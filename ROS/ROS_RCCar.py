"""ROS RC Car Domain"""

import sys, os
# Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domains import *
from ROS import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class ROS_RCCar(Domain):
    """
    This is a modified version of the original RC_Car domain.
    This version uses ROS and Mark Cutter's rc car simulation to calculate
    the step transitions. Most of it is the same, however some class variables,
    the __init__ function and the step function have been modified.
    """

    actions_num = 9
    state_space_dims = 4
    continuous_dims = arange(state_space_dims)

    ROOM_WIDTH = 3 # in meters
    ROOM_HEIGHT = 2 # in meters
    XMIN = -ROOM_WIDTH/2.0
    XMAX = ROOM_WIDTH/2.0
    YMIN = -ROOM_HEIGHT/2.0
    YMAX = ROOM_HEIGHT/2.0
    #ACCELERATION = .1
    TURN_ANGLE   = pi/6
    SPEEDMIN = -300
    SPEEDMAX = 300
    HEADINGMIN = -pi
    HEADINGMAX = pi
    INIT_STATE = array([0.0, 0.0, 0.0, 0.0])
    STEP_REWARD = -1
    GOAL_REWARD = 0
    GOAL        = [.5,.5]
    GOAL_RADIUS = .1
    actions = outer([-1, 0, 1],[-1, 0, 1])
    gamma = .9
    episodeCap = 10000
    #delta_t = .1 #time between steps
    CAR_LENGTH = .3 # L on the webpage
    CAR_WIDTH  = .15
    REAR_WHEEL_RELATIVE_LOC = .05 # The location of rear wheels if the car facing right with heading 0
    #Used for visual stuff:
    domain_fig          = None
    X_discretization    = 20
    Y_discretization    = 20
    SPEED_discretization = 5
    HEADING_discretization = 3
    ARROW_LENGTH        = .2
    car_fig             = None

    def __init__(self, noise = 0, logger = None):
        self.statespace_limits = array([[self.XMIN, self.XMAX], [self.YMIN, self.YMAX], [self.SPEEDMIN, self.SPEEDMAX], [self.HEADINGMIN, self.HEADINGMAX]])
        self.Noise = noise
        self.ros = RC_Com()
        super(ROS_RCCar,self).__init__(logger)

    def step(self, a):
        ns = self.ros.Step(self.state,a)
        self.state = ns.copy()
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal

    def s0(self):
        self.ros.resetState()
        self.state = self.INIT_STATE
        return self.state.copy()

    def isTerminal(self):
        return linalg.norm(self.state[0:2]-self.GOAL) < self.GOAL_RADIUS

