"""RC-Car domain"""

from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from .Domain import Domain
import numpy as np

__author__ = "Alborz Geramifard"


class RCCar(Domain):

    """
    This is a simple simulation of Remote Controlled Car in a room with no obstacle.

    **STATE:** 4 continuous dimensions:

    * x, y: (center point on the line connecting the back wheels),
    * speed (S on the webpage)
    * heading (theta on the webpage) w.r.t. body frame.
        positive values => turning right, negative values => turning left

    **ACTIONS:** Two action dimensions:

    * accel [forward, coast, backward]
    * phi [turn left, straight, turn Right]

    This leads to 3 x 3 = 9 possible actions.

    **REWARD:** -1 per step, 100 at goal.

    **REFERENCE:**

    .. seealso::
        http://planning.cs.uiuc.edu/node658.html

    """

    actions_num = 9
    state_space_dims = 4
    continuous_dims = np.arange(state_space_dims)

    ROOM_WIDTH = 3  # in meters
    ROOM_HEIGHT = 2  # in meters
    XMIN = -ROOM_WIDTH / 2.0
    XMAX = ROOM_WIDTH / 2.0
    YMIN = -ROOM_HEIGHT / 2.0
    YMAX = ROOM_HEIGHT / 2.0
    ACCELERATION = .1
    TURN_ANGLE = np.pi / 6
    SPEEDMIN = -.3
    SPEEDMAX = .3
    HEADINGMIN = -np.pi
    HEADINGMAX = np.pi
    INIT_STATE = np.array([0.0, 0.0, 0.0, 0.0])
    STEP_REWARD = -1
    GOAL_REWARD = 0
    GOAL = [.5, .5]
    GOAL_RADIUS = .1
    actions = np.outer([-1, 0, 1], [-1, 0, 1])
    discount_factor = .9
    episodeCap = 10000
    delta_t = .1  # time between steps
    CAR_LENGTH = .3  # L on the webpage
    CAR_WIDTH = .15
    # The location of rear wheels if the car facing right with heading 0
    REAR_WHEEL_RELATIVE_LOC = .05
    # Used for visual stuff:
    domain_fig = None
    X_discretization = 20
    Y_discretization = 20
    SPEED_discretization = 5
    HEADING_discretization = 3
    ARROW_LENGTH = .2
    car_fig = None

    def __init__(self, noise=0):
        self.statespace_limits = np.array(
            [[self.XMIN,
              self.XMAX],
             [self.YMIN,
              self.YMAX],
                [self.SPEEDMIN,
                 self.SPEEDMAX],
                [self.HEADINGMIN,
                 self.HEADINGMAX]])
        self.noise = noise
        super(RCCar, self).__init__()

    def step(self, a):
        x, y, speed, heading = self.state
        # Map a number between [0,8] to a pair. The first element is
        # acceleration direction. The second one is the indicator for the wheel
        acc, turn = id2vec(a, [3, 3])
        acc -= 1                # Mapping acc to [-1, 0 1]
        turn -= 1                # Mapping turn to [-1, 0 1]

        # Calculate next state
        nx = x + speed * np.cos(heading) * self.delta_t
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        # Bound values
        nx = bound(nx, self.XMIN, self.XMAX)
        ny = bound(ny, self.YMIN, self.YMAX)
        nspeed = bound(nspeed, self.SPEEDMIN, self.SPEEDMAX)
        nheading = wrap(nheading, self.HEADINGMIN, self.HEADINGMAX)

        # Collision to wall => set the speed to zero
        if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
            nspeed = 0

        ns = np.array([nx, ny, nspeed, nheading])
        self.state = ns.copy()
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.INIT_STATE.copy()
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        return np.linalg.norm(self.state[0:2] - self.GOAL) < self.GOAL_RADIUS

    def showDomain(self, a):
        s = self.state
        # Plot the car
        x, y, speed, heading = s
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure()
            # Goal
            plt.gca(
            ).add_patch(
                plt.Circle(
                    self.GOAL,
                    radius=self.GOAL_RADIUS,
                    color='g',
                    alpha=.4))
            plt.xlim([self.XMIN, self.XMAX])
            plt.ylim([self.YMIN, self.YMAX])
            plt.gca().set_aspect('1')
        # Car
        if self.car_fig is not None:
            plt.gca().patches.remove(self.car_fig)

        self.car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            self.CAR_LENGTH,
            self.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + plt.gca().transData
        self.car_fig.set_transform(rotation)
        plt.gca().add_patch(self.car_fig)

        plt.draw()
