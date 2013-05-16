#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys, os
# Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *
from ROS import *

#######################################################################
# \author Developed by Alborz Geramifard March 14th 2013 at MIT, modified by Elliott Williams
#######################################################################
# This is a modified version of the original RC_Car domain. This version uses ROS and Mark Cutter's rc car simulation to calculate the step transitions.
# Most of it is the same, however some class variables, the __init__ function and the step function have been modified. 

class ROS_RCCar(Domain):
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
        super(RCCar,self).__init__(logger)
    def step(self, s, a):
        x,y,speed,heading       = s
        acc,turn                = id2vec(a,[3,3])   #Map a number between [0,8] to a pair. The first element is acceleration direction. The second one is the indicator for the wheel
        acc                     -= 1                # Mapping acc to [-1, 0 1]
        turn                    -= 1                # Mapping turn to [-1, 0 1]
        
        #Calculate next state
        nx          = x + speed*cos(heading)*self.delta_t
        ny          = y + speed*sin(heading)*self.delta_t
        nspeed      = speed + acc*self.ACCELERATION*self.delta_t    
        nheading    = heading + speed/self.CAR_LENGTH*tan(turn*self.TURN_ANGLE)
        
        #Bound values
        nx          = bound(nx,self.XMIN,self.XMAX)
        ny          = bound(ny,self.YMIN,self.YMAX)
        nspeed      = bound(nspeed,self.SPEEDMIN,self.SPEEDMAX)
        nheading    = wrap(nheading,self.HEADINGMIN, self.HEADINGMAX)

    def step(self, s, a):
        ns = self.ros.Step(s,a)
        terminal = self.isTerminal(ns)
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal 

    def s0(self):
	self.ros.resetState()
        return self.INIT_STATE
    def isTerminal(self,s):
        return linalg.norm(s[0:2]-self.GOAL) < self.GOAL_RADIUS
    def showDomain(self, s, a):
        # Plot the car 
        x,y,speed,heading = s
        car_xmin = x-self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y-self.CAR_WIDTH/2.
        if self.domain_fig == None: # Need to initialize the figure
            self.domain_fig = pl.figure()
            #Goal
            pl.gca().add_patch(pl.Circle(self.GOAL, radius = self.GOAL_RADIUS, color = 'g', alpha= .4))
            pl.xlim([self.XMIN, self.XMAX])
            pl.ylim([self.YMIN, self.YMAX])
            pl.gca().set_aspect('1')
        #Car
        if self.car_fig != None:
            pl.gca().patches.remove(self.car_fig)

        self.car_fig = mpatches.Rectangle([car_xmin, car_ymin], self.CAR_LENGTH, self.CAR_WIDTH,alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(x,y,heading*180/pi) + pl.gca().transData
        self.car_fig.set_transform(rotation) 
        pl.gca().add_patch(self.car_fig)

        pl.draw()
            
if __name__ == '__main__':
    # p = GridWorld('/Domains/GridWorldMaps/ACC2011.txt');
    p = RCCar();
    p.test(10000)
