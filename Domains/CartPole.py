import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
#######################################################
# Robert Klein, Alborz Geramifard at MIT, Nov. 30 2012#
#######################################################
# State is angular position of pendulum on [0, 2*pi),
# (relative to straight down at 0 rad),and
# Angular rate of pendulum, [-15*pi, 15*pi].
# State space is discretized into uniform intervals, default
# 20 intervals each, for a state space size of 20^2=400.
# Actions take the form of torque applied directly
# (STANDS IN CONTRAST TO THE 4-STATE SYSTEM CARTPOLE);
# [-1, 0, 1] are the default available actions.
#
# The objective is to keep the pendulum in the goal
# region for as long as possible; default goal
# region is [pi - pi/6, pi + pi/6], no reward is
# obtained anywhere else.
# Optional noise is added in the form of torques on the
# pendulum.
######################################################

AVAIL_TORQUES = [-1,0,1] # Newtons, N - Torque values available as actions
GOAL_REGION = [pi - pi/6, pi + pi/6] # radians, rad - Goal region,
GOAL_REWARD = 1 # Reward obtained for remaining in the goal region
# Implicit 0 reward everywhere else - it's all relative.
ANGLE_LIMITS = [0, 2*pi] # rad - Limits on pendulum angle (NOTE we wrap the angle at 2*pi)
ANGLE_RATE_LIMITS=  [-15*pi, 15*pi] # rad / s
start_angle = 0 # rad - Starting angle of the pendulum
start_angle_rate = 0 # rad/s - Starting rate of the pendulum
NUM_ANGLE_INTERVALS = 20 # Number of discrete states in the angular range
NUM_ANGLE_RATE_INTERVALS = 20 # Number of discrete states in the angular velocity
NUM_CART_INTERVALS = 20 # Number of cart position intervals
NUM_CART_VEL_INTERVALS = 20 # Number cart velocity intervals
MASS_BOB = 1.0 # kilograms, kg - Mass of the bob at the end of the pendulum (assume zero arm mass)
MASS_CART = 1.0 # kg - Mass of the cart
LENGTH = 1.0 # meters, m - Length of the pendulum, meters
ACCEL_G = 9.81 # m/s^2 - gravitational constant

dt = 0.02 # Time between steps

ang_rate_noise = 0 # Noise (in rad/s) of pendulum angular rate

EPISODE_CAP = 200 # 200 used in tutorial

cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers
# are constrained by the format expected by ode functions.

class CartPole(Domain):
    def __init__(self, ang_rate_noise = 0, start_angle = 0, start_rate = 0, dt = 0.02):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.state_space_limits = [self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS]
        self.states_num = NUM_ANGLE_INTERVALS * NUM_RATE_INTERVALS       # Number of states
        self.actions_num = len(AVAIL_TORQUES)      # Number of Actions
        state_space_dims = 2 # Number of dimensions of the state space
        self.episodeCap = EPISODE_CAP       # The cap used to bound each episode (return to s0 after)
        self.start_angle = start_angle
        self.start_rate = start_rate
        self.dt = dt
        
    def showDomain(self,s,a = 0):
        pass
    def showLearning(self,representation):
        pass
    def s0(self):       
        # Returns the initial state
        return [self.start_angle, self.start_rate]
    def possibleActions(self,s):
        return self.AVAIL_TORQUES
    def step(self,s,a):
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
        # Simulate one step of the pendulum after taking torque action a
        # Adapted from pybrain code for cartpole
        self.curAction = a # store action so it can be retrieved by other methods.
        # Unfortunate that scipy ode methods do not allow additional parameters to be passed.
        
        ns = integrate.odeint(_dsdt, s, [0, self.dt])
        print 'Before -1',ns # why do we need to do ns[-1]
        ns = ns[-1]
        print 'After -1',ns
        return ns, self._earnedReward(ns), self.NOT_TERMINATED
        
    # From pybrain environment 'cartpole' 
def _dsdt(self, s, t):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # of the state, which can then 
        a = self.curAction
        (theta, thetaDot, x, s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        mp = self.mp
        mc = self.mc
        l = self.l
        u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
        v = s_
        v_ = (F - mp * l * (u_ * cos_theta - (s_ ** 2 * sin_theta))) / (mc + mp)
        return (u, u_, v, v_)
    def _earnedReward(self, s):
        if(GOAL_REGION[0] < s[StateIndex.THETA] < GOAL_REGION[1]):
            return GOAL_REWARD
        else: return 0
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        return self.NOT_TERMINATED # Pendulum has no absorbing state
    
    
    random.seed(0)