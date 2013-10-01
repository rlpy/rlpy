"""Classic mountain car task."""

from Tools import *
from Domain import Domain
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Josh Joseph", "Alborz Geramifard"]


class MountainCar(Domain):
    """
    based on
    http://library.rl-community.org/wiki/Mountain_Car_(Java)
    """
    actions_num = 3
    state_space_dims = 2
    continuous_dims = [0,1]

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
    #gamma = .9
    episodeCap = 10000

    #Used for visual stuff:
    domain_fig          = None
    valueFunction_fig   = None
    policy_fig          = None
    actionArrow         = None
    X_discretization    = 20
    XDot_discretization = 20
    CAR_HEIGHT          = .2
    CAR_WIDTH           = .1
    ARROW_LENGTH        = .2
    def __init__(self, noise = 0, logger = None):
        self.statespace_limits = array([[self.XMIN, self.XMAX], [self.XDOTMIN, self.XDOTMAX]])
        self.Noise = noise
        #Visual stuff:
        self.xTicks         = linspace(0,self.X_discretization-1,5)
        self.xTicksLabels   = linspace(self.XMIN,self.XMAX,5)
        self.yTicks         = linspace(0,self.XDot_discretization-1,5)
        self.yTicksLabels   = linspace(self.XDOTMIN,self.XDOTMAX,5)
        self.MIN_RETURN     = self.STEP_REWARD*(1-self.gamma**self.episodeCap)/(1-self.gamma) if self.gamma != 1 else self.STEP_REWARD*self.episodeCap
        self.MAX_RETURN     = 0
        self.DimNames       = ['X','Xdot']
        super(MountainCar,self).__init__(logger)

    def step(self, a):
        position, velocity = self.state
        noise = 2 * self.accelerationFactor * self.noise * (self.random_state.rand() - .5)
        velocity += (noise +
                                self.actions[a] * self.accelerationFactor +
                                cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity
        position = bound(position, self.XMIN, self.XMAX)
        if position < self.XMIN and velocity < 0: velocity = 0  # Bump into wall
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        ns = array([position, velocity])
        self.state = ns.copy()
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.INIT_STATE.copy()
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        return self.state[0] > self.GOAL

    def showDomain(self, a):
        s = self.state
        # Plot the car and an arrow indicating the direction of accelaration
        # Parts of this code was adopted from Jose Antonio Martin H. <jamartinh@fdi.ucm.es> online source code
        pos,vel = s
        if self.domain_fig == None: # Need to initialize the figure
            self.domain_fig = plt.figure("Mountain Car Domain")
            #plot mountain
            mountain_x = linspace(self.XMIN,self.XMAX,1000)
            mountain_y = sin(3*mountain_x);
            plt.gca().fill_between( mountain_x, min(mountain_y)-self.CAR_HEIGHT*2,mountain_y, color='g' )
            plt.xlim([self.XMIN-.2,self.XMAX])
            plt.ylim([min(mountain_y)-self.CAR_HEIGHT*2,max(mountain_y)+self.CAR_HEIGHT*2])
            #plot car
            self.car = lines.Line2D([],[], linewidth=20, color='b',alpha=.8)
            plt.gca().add_line(self.car)
            #Goal
            plt.plot(self.GOAL, sin(3*self.GOAL),'yd',markersize=10.0)
            plt.axis('off')
            plt.gca().set_aspect('1')
        self.domain_fig = plt.figure("Mountain Car Domain")
        #pos = 0
        #a = 0
        car_middle_x    = pos
        car_middle_y    = sin(3*pos)
        slope           = arctan(3*cos(3*pos))
        car_back_x      = car_middle_x - self.CAR_WIDTH*cos(slope)/2.
        car_front_x     = car_middle_x + self.CAR_WIDTH*cos(slope)/2.
        car_back_y      = car_middle_y - self.CAR_WIDTH*sin(slope)/2.
        car_front_y     = car_middle_y + self.CAR_WIDTH*sin(slope)/2.
        self.car.set_data([car_back_x, car_front_x], [car_back_y, car_front_y])
        #wheels
        #plott(x(1)-0.05,sin(3*(x(1)-0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);
        #plot(x(1)+0.05,sin(3*(x(1)+0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);
        # Arrows
        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None

        SHIFT = .1
        if self.actions[a] > 0:
            self.actionArrow = fromAtoB(
                                          car_front_x , car_front_y,
                                          car_front_x +self.ARROW_LENGTH*cos(slope), car_front_y+self.ARROW_LENGTH*sin(slope),
                                          #car_front_x + self.CAR_WIDTH*cos(slope)/2., car_front_y + self.CAR_WIDTH*sin(slope)/2.+self.CAR_HEIGHT,
                                          'k',"arc3,rad=0",
                                          0,0, 'simple'
                                          )
        if self.actions[a] < 0:
            self.actionArrow = fromAtoB(
                                          car_back_x , car_back_y,
                                          car_back_x - self.ARROW_LENGTH*cos(slope), car_back_y-self.ARROW_LENGTH*sin(slope),
                                          #car_front_x + self.CAR_WIDTH*cos(slope)/2., car_front_y + self.CAR_WIDTH*sin(slope)/2.+self.CAR_HEIGHT,
                                          'r',"arc3,rad=0",
                                          0,0, 'simple'
                                          )
        plt.draw()


    def showLearning(self,representation):
        pi      = zeros((self.X_discretization, self.XDot_discretization),'uint8')
        V       = zeros((self.X_discretization, self.XDot_discretization))

        if self.valueFunction_fig is None:
            self.valueFunction_fig   = plt.figure("Value Function")
            self.valueFunction_im   = plt.imshow(V, cmap='ValueFunction',interpolation='nearest',origin='lower',vmin=self.MIN_RETURN,vmax=self.MAX_RETURN)

            plt.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            plt.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\dot x$")

            self.policy_fig = plt.figure("Policy")
            self.policy_im = plt.imshow(pi, cmap='MountainCarActions', interpolation='nearest',origin='lower',vmin=0,vmax=self.actions_num)

            plt.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            plt.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\dot x$")
            plt.show()

        for row, xDot in enumerate(linspace(self.XDOTMIN, self.XDOTMAX, self.XDot_discretization)):
            for col, x in enumerate(linspace(self.XMIN, self.XMAX, self.X_discretization)):
                s           = [x,xDot]
                Qs       = representation.Qs(s, False)
                As       = self.possibleActions()
                pi[row,col] = representation.bestAction(s, False, As)
                V[row,col]  = max(Qs)
        self.valueFunction_im.set_data(V)
        self.policy_im.set_data(pi)

        self.valueFunction_fig   = plt.figure("Value Function")
        plt.draw()
        self.policy_fig = plt.figure("Policy")
        plt.draw()
