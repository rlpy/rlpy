import numpy as np
from Domain import Domain

XMIN = -3.7
XMAX = 0.5
XDOTMIN = -0.07
XDOTMAX = 0.07
INITSTATE = np.array([-np.pi / 2.0 / 3.0, 0.0])

class Mountaincar(Domain):
    actions_num = 2
    state_space_dims = 2
    episodeCap = 500

    def __init__(self):
        self.actions = np.array([-1, 1])
        self.bounds = np.array([[XMIN, XMAX],[XDOTMIN, XDOTMAX]]).transpose()
        self.start = INITSTATE

    def step(self, x, u):
        x_next = x.copy()
        x_next[0] = min(max(x[0]+x[1], self.bounds[0,0]), self.bounds[1,0])
        x_next[1] = min(max(x[1]+0.001*u+(-0.0025*np.cos(3*x[0])), self.bounds[0,1]), self.bounds[1,1])
        return x_next

    def s0(self):
        return self.start

    def possibleActions(self, x):
        return self.actions