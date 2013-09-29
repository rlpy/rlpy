from Tools import *
from Domain import Domain

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
            "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Robert H. Klein", "Alborz Geramifard"]


class PST(Domain):
    """
    Persistent Search and Track Mission per [MATLAB implementation] \n

    --------------------DESCRIPTION----------------------- \n
    Each UAV has 4 states; location, fuel, actuator status, sensor status.

    There are 4 types of locations a uav can be in; Base,
    Refuel, Communications, surveillance. \n
    Loitering for 1 timestep at Refuel assigns fuel of 10 to that UAV.\n
    Loitering for 1 timestep at Base assigns status 1 to
    Actuator and Sensor.

    Goal is to maintain as many UAVS with working sensor in the Surveillance
    state as there are targets (NUM_TARGETS), while maintaining at least 1 UAV
    in each communication state. \n
    Reward of 20 is earned for each UAV in the surveillance state with working
    sensor, up to the number of targets. \n
    If no UAV is in the communication state, zero reward earned for surveillance.
    (No reward is earned for excess UAVs or those with failed sensor)

    If the actuator fails, the UAV can only take actions leading it back to the
    refuel or base states, where it may loiter.

    A penalty is applied for each unit of fuel consumed,
    which occurs when a UAV moves between locations or when it is loitering
    above a communications or surveilllance location.
    (ie, no penalty when loitering at REFUEL or BASE)

    Finally, if any UAV has fuel 0, the episode terminates with large penalty.

    --------------------FORMULATION----------------------- \n
    State vector consists of 4 blocks of states,
    each corresponding to a property of the UAVs
    [locations, fuel quantities, actuator statuses, sensor statuses] \n
    So for example: \n
    state [1,2,9,3,1,0,1,1] -->
    [1,2] | [9,3] | [1,0] | [1,1] --> 2 UAVs:
    UAV1 in location 1, with 9 fuel units remaining, and
    sensor + actuator with status 1 (functioning). \n
    UAV 2 in location 2, 3 fuel units remaining, actuator
    with status 0 and sensor with status 1.

    Location transitions, sensor and actuator failures, and
    fuel consumption are stochastic.


    --------------------VISUALIZATION----------------------- \n
    In the current visualization, the 'actuator' which enables communication
    is represented by a wedge above each UAV circle; a failed actuator is red,
    functional is black.  Similarly, the 'sensor' which enables surveillance
    is represented by a wedge in front of the UAV circle, with similar coloring.

    If each comms location has at least 1 UAV able to perform communications,
    lines are drawn connecting the surveillance location through each comms
    location back to base. \n
    If at least 1 vehicle is actively performing surveillance, this line is black; \n
    Else If there are no vehicles performing surveillance, this line is red. \n
    Else if there is no available comms link, no line is drawn. \n
    Else If it least one capable vehicle is in the surveillance region but no comms
    link is available, a partial red comms line is drawn with a red vertical
    line cutting down its center.

    """
    episodeCap          = 1000 # 100 used in tutorial, 1000 in matlab
    gamma               = 0.9  # 0.9 used in tutorial and matlab

    # Domain constants
    FULL_FUEL           = 10   # Number of fuel units at start [10 in tutorial]
    P_ACT_FAIL          = 0.05 # Probability that actuators fail on this timestep [0.02 in tutorial]
    P_SENSOR_FAIL       = 0.05 # Probability that sensors fail on this timestep [0.05 in tutorial]
#    CRASH_REWARD_COEFF  = -2.0 # Negative reward coefficient for running out of fuel (applied on every step) [C_crash] [-2.0 in tutorial]
    CRASH_REWARD        = -50
    SURVEIL_REWARD      = 20 # Positive reward coefficient for performing surveillance on each step [C_cov] [1.5 in tutorial]
    FUEL_BURN_REWARD_COEFF = -1 # Negative reward coefficient: for fuel burn penalty [not mentioned in MDP Tutorial]
    MOVE_REWARD_COEFF   = 0   # Reward (negative) coefficient for movement (i.e., fuel burned while loitering might be penalized above, but no movement cost)
    NUM_TARGET          = 1   # Number of targets; SURVEIL_REWARD is multiplied by the number of targets successfully observed
    NUM_UAV             = None # Number of UAVs present in the mission [3 in tutorial
    NOM_FUEL_BURN       = 1 # Nominal rate of fuel depletion selected with probability P_NOM_FUEL_BURN

    # Domain variables
    motionNoise         = 0    # Noise in action (with some probability, loiter rather than move)
    numCrashed          = 0    # Number of crashed UAVs [n_c]
    numHealthySurveil   = 0    # Number of UAVs in surveillance area with working sensor and actuator [n_s]
    fuelUnitsBurned     = 0
    LIMITS              = []   # Limits on action indices
    isCommStatesCovered = False # All comms states are covered on a given timestep, enabling surveillance rewards

    # Plotting constants
    UAV_RADIUS = 0.3
    SENSOR_REL_X        = 0.2 # Location of the sensor image relative to the uav
    SENSOR_LENGTH       = 0.2 # Length of the sensor surveillance image
    ACTUATOR_REL_Y      = 0.2 # Location of the actuator image relative to the uav
    ACTUATOR_HEIGHT     = 0.2 # Height of the actuator comms image
    domain_fig          = None
    subplot_axes        = None
    location_rect_vis   = None # List of rectangle objects used in the plot
    uav_circ_vis        = None # List of circles used to represent UAVs in plot
    uav_text_vis        = None # List of fuel text used in plot
    uav_sensor_vis      = None # List of sensor wedges used in plot
    uav_actuator_vis    = None # List of actuator wedges used in plot
    comms_line          = None # List of communication lines used in plot
    location_coord      = None # Coordinates of the center of each rectangle
#    uav_vis_list = None # List of UAV objects used in plot
    LOCATION_WIDTH      = 1.0 # Width of each rectangle used to represent a location
    RECT_GAP            = 0.9   # Gap to leave between rectangles
    dist_between_locations = 0 # Total distance between location rectangles in plot, computed in init()

    ###
    def __init__(self, NUM_UAV = 3, motionNoise = 0, logger = None):
        self.NUM_UAV                = NUM_UAV
        self.states_num             = NUM_UAV * UAVIndex.SIZE       # Number of states (LOC, FUEL...) * NUM_UAV
        self.actions_num            = pow(UAVAction.SIZE,NUM_UAV)    # Number of Actions: ADVANCE, RETREAT, LOITER
        locations_lim = array(tile([0,UAVLocation.SIZE-1],(NUM_UAV,1)))
        fuel_lim =      array(tile([0,self.FULL_FUEL],(NUM_UAV,1)))
        actuator_lim =  array(tile([0,ActuatorState.SIZE-1],(NUM_UAV,1)))
        sensor_lim =    array(tile([0,SensorState.SIZE-1],(NUM_UAV,1)))
        self.statespace_limits      = vstack([locations_lim, fuel_lim, actuator_lim, sensor_lim])# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.motionNoise            = motionNoise # with some noise, when uav desires to transition to new state, remains where it is (loiter)
        self.LIMITS                 = UAVAction.SIZE * ones(NUM_UAV, dtype='int') # eg [3,3,3,3], number of possible actions
        self.isCommStatesCovered    = False # Don't have communications available yet, so no surveillance reward allowed.
        self.location_rect_vis      = None #
        self.location_coord         = None
        self.uav_circ_vis           = None
        self.uav_text_vis           = None
        self.uav_sensor_vis         = None
        self.uav_actuator_vis       = None
        self.comms_line             = None
        self.dist_between_locations = self.RECT_GAP + self.LOCATION_WIDTH
        self.DimNames = []
        [self.DimNames.append('UAV%d-loc' % i) for i in arange(NUM_UAV)]
        [self.DimNames.append('UAV%d-fuel' % i) for i in arange(NUM_UAV)]
        [self.DimNames.append('UAV%d-act' % i) for i in arange(NUM_UAV)]
        [self.DimNames.append('UAV%d-sen' % i) for i in arange(NUM_UAV)]
        super(PST,self).__init__(logger)
        if self.logger: self.logger.log("NUM_UAV:\t\t%d" % self.NUM_UAV)

    def resetLocalVariables(self):
        self.numCrashed = 0 # Number of crashed UAVs [n_c]


    def showDomain(self, a=0):
        s = self.state
        if self.domain_fig is None:
            self.domain_fig = pl.figure(1, (UAVLocation.SIZE * self.dist_between_locations + 1, self.NUM_UAV + 1))
            pl.show()
        pl.clf()
         #Draw the environment
         # Allocate horizontal 'lanes' for UAVs to traverse

        # Formerly, we checked if this was the first time plotting; wedge shapes cannot be removed from
        # matplotlib environment, nor can their properties be changed, without clearing the figure
        # Thus, we must redraw the figure on each timestep
        #        if self.location_rect_vis is None:
        # Figure with x width corresponding to number of location states, UAVLocation.SIZE
        # and rows (lanes) set aside in y for each UAV (NUM_UAV total lanes).  Add buffer of 1
        self.subplot_axes = self.domain_fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
        crashLocationX = 2*(self.dist_between_locations)*(UAVLocation.SIZE-1)
        self.subplot_axes.set_xlim(0, 1 + crashLocationX + self.RECT_GAP)
        self.subplot_axes.set_ylim(0, 1 + self.NUM_UAV)
        self.subplot_axes.xaxis.set_visible(False)
        self.subplot_axes.yaxis.set_visible(False)

        # Assign coordinates of each possible uav location on figure
        self.location_coord = [0.5 + (self.LOCATION_WIDTH / 2) + (self.dist_between_locations)*i for i in range(UAVLocation.SIZE-1)]
        self.location_coord.append(crashLocationX + self.LOCATION_WIDTH / 2)

         # Create rectangular patches at each of those locations
        self.location_rect_vis = [mpatches.Rectangle([0.5 + (self.dist_between_locations)*i, 0], self.LOCATION_WIDTH, self.NUM_UAV * 2, fc = 'w') for i in range(UAVLocation.SIZE-1)]
        self.location_rect_vis.append(mpatches.Rectangle([crashLocationX, 0], self.LOCATION_WIDTH, self.NUM_UAV * 2, fc = 'w'))
        [self.subplot_axes.add_patch(self.location_rect_vis[i]) for i in range(4)]
        self.comms_line = [lines.Line2D([0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*i, 0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*i + self.RECT_GAP],[self.NUM_UAV*0.5 + 0.5, self.NUM_UAV * 0.5 + 0.5], linewidth = 3, color='black', visible=False) for i in range(UAVLocation.SIZE-2)]
        self.comms_line.append(lines.Line2D([0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*2, crashLocationX],[self.NUM_UAV*0.5 + 0.5, self.NUM_UAV * 0.5 + 0.5], linewidth = 3, color='black', visible=False))

        # Create location text below rectangles
        locText = ["Base","Refuel","Communication","Surveillance"]
        self.location_rect_txt = [pl.text(0.5 + self.dist_between_locations*i + 0.5*self.LOCATION_WIDTH,-0.3,locText[i], ha = 'center') for i in range(UAVLocation.SIZE-1)]
        self.location_rect_txt.append(pl.text(crashLocationX + 0.5*self.LOCATION_WIDTH,-0.3,locText[UAVLocation.SIZE-1], ha = 'center'))

        # Initialize list of circle objects

        uav_x = self.location_coord[UAVLocation.BASE]

        # Update the member variables storing all the figure objects
        self.uav_circ_vis = [mpatches.Circle((uav_x,1+uav_id), self.UAV_RADIUS, fc="w") for uav_id in range(0,self.NUM_UAV)]
        self.uav_text_vis = [None for uav_id in range(0,self.NUM_UAV)] # fuck
        self.uav_sensor_vis = [mpatches.Wedge((uav_x+self.SENSOR_REL_X, 1+uav_id),self.SENSOR_LENGTH, -30, 30) for uav_id in range(0,self.NUM_UAV)]
        self.uav_actuator_vis =[mpatches.Wedge((uav_x, 1+uav_id + self.ACTUATOR_REL_Y),self.ACTUATOR_HEIGHT, 60, 120) for uav_id in range(0,self.NUM_UAV)]

        # The following was executed when we used to check if the environment needed re-drawing: see above.
                # Remove all UAV circle objects from visualization
        #        else:
        #            [self.uav_circ_vis[uav_id].remove() for uav_id in range(0,self.NUM_UAV)]
        #            [self.uav_text_vis[uav_id].remove() for uav_id in range(0,self.NUM_UAV)]
        #            [self.uav_sensor_vis[uav_id].remove() for uav_id in range(0,self.NUM_UAV)]


        # For each UAV:
        # Draw a circle, with text inside = amt fuel remaining
        # Triangle on top of UAV for comms, black = good, red = bad
        # Triangle in front of UAV for surveillance

        sStruct = self.state2Struct(s)

        for uav_id in range(0,self.NUM_UAV):
            # Assign all the variables corresponding to this UAV for this iteration;
            # this could alternately be done with a UAV class whose objects keep track
            # of these variables.  Elect to use lists here since ultimately the state
            # must be a vector anyway.
            uav_location = sStruct.locations[uav_id] # State index corresponding to the location of this uav
            uav_fuel = sStruct.fuel[uav_id]
            uav_sensor = sStruct.sensor[uav_id]
            uav_actuator = sStruct.actuator[uav_id]

            # Assign coordinates on figure where UAV should be drawn
            uav_x = self.location_coord[uav_location]
            uav_y = 1 + uav_id

            # Update plot wit this UAV
            self.uav_circ_vis[uav_id] = mpatches.Circle((uav_x,uav_y), self.UAV_RADIUS, fc="w")
            self.uav_text_vis[uav_id] = pl.text(uav_x-0.05, uav_y-0.05, uav_fuel)
            if uav_sensor == SensorState.RUNNING: objColor = 'black'
            else: objColor = 'red'
            self.uav_sensor_vis[uav_id] = mpatches.Wedge((uav_x+self.SENSOR_REL_X,uav_y),self.SENSOR_LENGTH, -30, 30, color=objColor)

            if uav_actuator == ActuatorState.RUNNING: objColor = 'black'
            else: objColor = 'red'
            self.uav_actuator_vis[uav_id] = mpatches.Wedge((uav_x,uav_y + self.ACTUATOR_REL_Y),self.ACTUATOR_HEIGHT, 60, 120, color=objColor)

            self.subplot_axes.add_patch(self.uav_circ_vis[uav_id])
            self.subplot_axes.add_patch(self.uav_sensor_vis[uav_id])
            self.subplot_axes.add_patch(self.uav_actuator_vis[uav_id])

        numHealthySurveil = sum(logical_and(sStruct.locations == UAVLocation.SURVEIL, sStruct.sensor))
        if (any(sStruct.locations == UAVLocation.COMMS)): # We have comms coverage: draw a line between comms states to show this
            [self.comms_line[i].set_visible(True) for i in range(len(self.comms_line))]
            [self.comms_line[i].set_color('black') for i in range(len(self.comms_line))]
            if numHealthySurveil > 0: # We also have UAVs in surveillance; color the comms line black
                self.location_rect_vis[len(self.location_rect_vis)-1].set_color('green')
                #self.comms_line[len(self.comms_line)-1].set_color('red')
                #self.comms_line[len(self.comms_line)-1].set_visible(True)
                #midpoint = (0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*2 + crashLocationX)/2
                #self.subplot_axes.add_line(lines.Line2D([midpoint, midpoint],[self.NUM_UAV + 0.75, self.NUM_UAV + 0.25], linewidth = 3, color='red', visible=True))
        [self.subplot_axes.add_line(self.comms_line[i]) for i in range(len(self.comms_line))] # Only visible lines actually appear
        pl.draw()
        sleep(0.5)

    def showLearning(self,representation):
        pass

    def step(self,a):
        # Note below that we pass the structure by reference to save time; ie, components of sStruct refer directly to s
        ns = self.state.copy()
        sStruct = self.state2Struct(self.state)
        nsStruct = self.state2Struct(ns)
        # Subtract 1 below to give -1,0,1, easily sum actions
        actionVector = array(id2vec(a,self.LIMITS)) # returns list of form [0,1,0,2] corresponding to action of each uav
        nsStruct.locations += (actionVector-1)

        #TODO - incorporate cost graph as in matlab.
        fuelBurnedBool = [(actionVector[i] == UAVAction.LOITER and (nsStruct.locations[i] == UAVLocation.REFUEL or nsStruct.locations[i] == UAVLocation.BASE)) for i in arange(self.NUM_UAV)]
        fuelBurnedBool = array(fuelBurnedBool) == False
        nsStruct.fuel = array([sStruct.fuel[i] - self.NOM_FUEL_BURN * fuelBurnedBool[i] for i in arange(self.NUM_UAV)]) # if fuel
        self.fuelUnitsBurned = sum(fuelBurnedBool)
        distanceTraveled = sum(logical_and(nsStruct.locations, sStruct.locations))

        # Actuator failure transition
        randomFails = array([self.random_state.random_sample() for dummy in arange(self.NUM_UAV)])
        randomFails = randomFails > self.P_ACT_FAIL
        nsStruct.actuator = logical_and(sStruct.actuator, randomFails)

        # Sensor failure transition
        randomFails = array([self.random_state.random_sample() for dummy in arange(self.NUM_UAV)])
        randomFails = randomFails > self.P_SENSOR_FAIL
        nsStruct.sensor = logical_and(sStruct.sensor, randomFails)

        #Refuel those in refuel node
        refuelIndices = nonzero(logical_and(sStruct.locations == UAVLocation.REFUEL, nsStruct.locations == UAVLocation.REFUEL))
        nsStruct.fuel[refuelIndices] = self.FULL_FUEL

        # Fix sensors and motors in base state
        baseIndices = nonzero(logical_and(sStruct.locations == UAVLocation.BASE, nsStruct.locations == UAVLocation.BASE))
        nsStruct.actuator[baseIndices] = ActuatorState.RUNNING
        nsStruct.sensor[baseIndices] = SensorState.RUNNING

        # Test if have communication
        self.isCommStatesCovered = any(sStruct.locations == UAVLocation.COMMS)

        surveillanceBool = (sStruct.locations == UAVLocation.SURVEIL)
        self.numHealthySurveil = sum(logical_and(surveillanceBool, sStruct.sensor))

        totalStepReward = 0

        ns = self.struct2State(nsStruct)
        self.state = ns.copy()

        ##### Compute reward #####
        if self.isCommStatesCovered:
            totalStepReward += self.SURVEIL_REWARD * min(self.NUM_TARGET, self.numHealthySurveil)
        if self.isTerminal(): totalStepReward += self.CRASH_REWARD
        totalStepReward += self.FUEL_BURN_REWARD_COEFF * self.fuelUnitsBurned + self.MOVE_REWARD_COEFF * distanceTraveled # Presently movement penalty is set to 0
        return totalStepReward,ns,self.isTerminal(), self.possibleActions()

    def s0(self):
        self.resetLocalVariables()
        locations   = ones(self.NUM_UAV, dtype='int') * UAVLocation.BASE
        fuel        = ones(self.NUM_UAV, dtype='int') * self.FULL_FUEL
        actuator    = ones(self.NUM_UAV, dtype='int') * ActuatorState.RUNNING
        sensor      = ones(self.NUM_UAV, dtype='int') * SensorState.RUNNING

        self.state = self.properties2StateVec(locations, fuel, actuator, sensor)
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def state2Struct(self,s):
        # Only perform multiplication once to save time
        fuelEndInd      = 2*self.NUM_UAV
        actuatorEndInd  = 3*self.NUM_UAV
        sensorEndInd    = 4*self.NUM_UAV
        locations = s[0         :self.NUM_UAV]
        fuel =      s[self.NUM_UAV   :fuelEndInd]
        actuator =  s[fuelEndInd :actuatorEndInd]
        sensor =    s[actuatorEndInd   :sensorEndInd]

        return StateStruct(locations, fuel, actuator, sensor)
    def properties2StateVec(self,locations, fuel, actuator, sensor):
        return hstack([locations, fuel, actuator, sensor])

    def struct2State(self,sState):
        return hstack([sState.locations, sState.fuel, sState.actuator, sState.sensor])

    def possibleActions(self):
        s = self.state
        # return the id of possible actions
        # find empty blocks (nothing on top)
        validActions = [] # Contains a list of uav_actions lists, e.g. [[0,1,2],[0,1],[1,2]] with the index corresponding to a uav.
        # First, enumerate the possible actions for each uav
        sStruct = self.state2Struct(s)
        for uav_id in range(0,self.NUM_UAV):
            uav_actions = []
            # Only allowed to loiter if have working actuator or are
            # in refuel/repair location
            if(sStruct.actuator[uav_id] == ActuatorState.RUNNING or \
            sStruct.locations[uav_id] == UAVLocation.REFUEL or \
            sStruct.locations[uav_id] == UAVLocation.BASE):
                uav_actions.append(UAVAction.LOITER)

            if(sStruct.fuel[uav_id] > 0): # This UAV is not crashed
                # Can advance to right as long as have working actuator and
                # are not in 'rightmost' state, surveillance
                if(sStruct.locations[uav_id] != UAVLocation.SURVEIL and \
                sStruct.actuator[uav_id] == ActuatorState.RUNNING):
                    uav_actions.append(UAVAction.ADVANCE)

                # Can retreat left anytime as long as aren't already in
                # 'leftmost' state, base
                if(sStruct.locations[uav_id] != UAVLocation.BASE):
                    uav_actions.append(UAVAction.RETREAT)
            else: # This UAV is crashed; give it a dummy action for now
                if(len(uav_actions) < 1): uav_actions.append(UAVAction.LOITER)
            if(isinstance(uav_actions,int)): # Test for single-UAV case
                validActions.append([uav_actions])
            else:
                validActions.append(uav_actions)
        return array(self.vecList2id(validActions, UAVAction.SIZE)) # TODO place this in tools

    # TODO place this in Tools
    # Given a list of lists of the form [[0,1,2],[0,1],[1,2],[0,1]]... return
    # unique id for each permutation between lists; eg above, would return 3*2*2*2 values
    # ranging from 0 to 3^4 -1 (3 is max value possible in each of the lists, maxValue)
    def vecList2id(self,x,maxValue):
    #returns a list of unique id's based on possible permutations of this list of lists
        _id = 0
        actionIDs = []
        curActionList = []
        lenX = len(x)
        limits = tile(maxValue, (1,lenX))[0] # eg [3,3,3,3] # TODO redundant computation
        self.vecList2idHelper(x,actionIDs,0, curActionList, maxValue,limits) # TODO remove self

        return actionIDs

    def vecList2idHelper(self,x,actionIDs,ind,curActionList, maxValue,limits):
    #returns a list of unique id's based on possible permutations of this list of lists.  See vecList2id
        for curAction in x[ind]: # x[ind] is one of the lists, e.g [0, 2] or [1,2]
            partialActionAssignment = curActionList[:]
            partialActionAssignment.append(curAction)
            if(ind == len(x) - 1): # We have reached the final list, assignment is complete
                actionIDs.append(vec2id(partialActionAssignment, limits)) # eg [0,1,0,2] and [3,3,3,3]
            else:
                self.vecList2idHelper(x,actionIDs,ind+1,partialActionAssignment, maxValue,limits) # TODO remove self
    def isTerminal(self):
        sStruct = self.state2Struct(self.state)
        return any(logical_and(sStruct.fuel <= 0, sStruct.locations != UAVLocation.REFUEL))

class UAVLocation:
    BASE = 0
    REFUEL      = 1
    COMMS       = 2
    SURVEIL     = 3
    SIZE        = 4

class StateStruct:
    def __init__(self, locations, fuel, actuator, sensor):
        self.locations = locations
        self.fuel       = fuel
        self.actuator   = actuator
        self.sensor     = sensor

class ActuatorState:
    FAILED, RUNNING = 0,1 # Status of actuator
    SIZE = 2
class SensorState:
    FAILED, RUNNING = 0,1 # Status of sensor
    SIZE = 2

# Future code (in 'step') assumes that these can be summed, ie, 0,1,2 retreat, advance, loiter
class UAVAction:
    ADVANCE,RETREAT,LOITER = 2,0,1 # Available actions
    SIZE = 3 # number of states above

# NOTE: properties2StateVec assumes the order loc,fuel,actuator,sensor
class UAVIndex:
    LOC, FUEL, ACT_STATUS, SENS_STATUS = 0,1,2,3
    SIZE = 4 # Number of indices required for state of each UAV






