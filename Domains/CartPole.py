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

class CartPole(Domain):
    AVAIL_TORQUES = [-20,0,20] # Newtons, N - Torque values available as actions
#    AVAIL_TORQUES = [-1,0,1] # Newtons, N - Torque values available as actions
    
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
    
    dt = 0.10 # Time between steps
    
    torque_noise_sigma = 0.1 # Noise std deviation (in N*m) of applied pendulum torque [0.1 in tutorial]
    
    EPISODE_CAP = 200 # 
    
    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers
#NOT IMPLEMENTED    s_continuous = None # Current actual, continuous state
    
    # Plotting variables
    pendulumArm = None
    domainFig = None
    subplotAxes = None
    CIRCLE_RADIUS = 0.2
    PENDULUM_PIVOT = [0,0] # pivot point of pendulum
    
    ## NOT IMPLEMENTED
