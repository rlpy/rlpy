import sys, os
# Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

#######################################################################
# Developed by Josh Joseph and Alborz Geramifard Nov 15th 2012 at MIT #
#######################################################################
# Mountain Car Domain Based on http://library.rl-community.org/wiki/Mountain_Car_(Java)


class MountainCar(Domain):
    actions_num = 3
    state_space_dims = 2
    episodeCap = 500
    XMIN = -1.2
    XMAX = 0.6
    XDOTMIN = -0.07
    XDOTMAX = 0.07
    INIT_STATE = array([-0.5, 0.0])
    STEP_REWARD = -1
    GOAL_REWARD = 0
    GOAL = .5
    actions = [-1, 0, 1] 
    noise = 0
    accelerationFactor = 0.001
    gravityFactor = -0.0025;
    hillPeakFrequency = 3.0;
    continous_dims = [0,1]
    def __init__(self, noise = 0):
        self.statespace_limits = array([[self.XMIN, self.XMAX], [self.XDOTMIN, self.XDOTMAX]])
        self.Noise = noise
        super(MountainCar,self).__init__()
    def step(self, s, a):
        position, velocity = s
        noise = 2 * self.accelerationFactor * self.noise * (random.rand() - .5) 
        velocity += (noise + 
                                self.actions[a] * self.accelerationFactor + 
                                cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity 
        position = bound(position, self.XMIN, self.XMAX)
        if position < self.XMIN and velocity < 0: velocity = 0  # Bump into wall
        terminal = self.isTerminal(s)
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        ns = array([position, velocity])        
        return r, ns, terminal 
    def s0(self):
        return self.INIT_STATE
    def showDomain(self, s, a):
        print s, a
    def isTerminal(self,s):
        return s[0] > self.GOAL
if __name__ == '__main__':
    # p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
    p = MountainCar();
    p.test(1000)
