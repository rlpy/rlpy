import sys, os

import csv
#



#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RL-Python/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RL-Python'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *
# \cond DEV

########################################################
# Robert H Klein, Alborz Geramifard Nov 26 2012 at MIT #
########################################################
# Persistent Search and Track Mission with:
# NUM_UAV:       num vehicles present
# NUM_COMMS  num intermediate communication states
# Other associated parameters shown below.
#
#   Goal is to maintain as many UAVs with functional
# sensor and actuator as possible in the
# surveillance state, while maintaining at least 1 UAV
# with working actuator in each communication state.
#
#   Failure to satisfy the communication state above
# results in zero reward earned for surveillance.
#
#   State vector consists of blocks of 4 states,
# corresponding to each UAV:
# [location, fuel qty, actuator status, sensor status]
#   so for example, state [1,9,1,1,2,3,0,1] -->
# [1,9,1,1 || 2,3,0,1] --> 2 UAV's;
#   UAV1 in location 1, with 9 fuel units remaining, and
# sensor + actuator with status 1 (defined in the classes below).
#   UAV 2 in location 2, 3 fuel units remaining, actuator
# with status 0 and sensor with status 1.
#
#   Location transitions, sensor and actuator failures, and
# fuel consumption are stochastic.
#
#
#   In the current visualization, the 'actuator' which enables communication
# is represented by a wedge above each UAV circle; a failed actuator is red,
# functional is black.  Similarly, the 'sensor' which enables surveillance
# is represented by a wedge in front of the UAV circle, with similar coloring.
#
# If each comms location has at least 1 UAV able to perform communications,
# lines are drawn connecting the surveillance location through each comms
# location back to base.
# If at least 1 vehicle is actively performing surveillance, this line is black;
# Else If there are no vehicles performing surveillance, this line is red.
# Else if there is no available comms link, no line is drawn.

# Else If it least one capable vehicle is in the surveillance region but no comms
# link is available, a partial red comms line is drawn with a red vertical
# line cutting down its center.
# 
########################################################
class UAVLocation:
    MAINTENANCE = 0
    REFUEL      = 1
    COMMS       = 2
    SURVEIL     = 3
    SIZE        = 4
    
class ActuatorState:
    FAILED, RUNNING = 0,1 # Status of actuator
    SIZE = 2
class SensorState:
    FAILED, RUNNING = 0,1 # Status of sensor
    SIZE = 2
class UAVAction:
    ADVANCE,RETREAT,LOITER = 2,0,1 # Available actions
    SIZE = 3 # number of states above

class UAVIndex:
    UAV_LOC, UAV_FUEL, UAV_ACT_STATUS, UAV_SENS_STATUS = 0,1,2,3
    SIZE = 4 # Number of indices required for state of each UAV
        
## @author Robert H. Klein
class PST(Domain):
    
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
    NOM_FUEL_BURN       = 1 # Nominal rate of fuel depletion (at present, there is no other rate)
    
    # Domain variables
    motionNoise         = 0    # Noise in action (with some probability, loiter rather than move)
    numCrashed          = 0    # Number of crashed UAVs [n_c]
    numHealthySurveil   = 0    # Number of UAVs in surveillance area with working sensor and actuator [n_s]
    fuelUnitsBurned     = 0
    LIMITS = []    
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
        self.states_num             = NUM_UAV * UAVIndex.SIZE       # Number of states (UAV_LOC, UAV_FUEL...) * NUM_UAV
        self.actions_num            = pow(UAVAction.SIZE,NUM_UAV)    # Number of Actions: ADVANCE, RETREAT, LOITER
        _statespace_limits = array(vstack([[0,UAVLocation.SIZE-1],[0,self.FULL_FUEL],[0,ActuatorState.SIZE-1],[0,SensorState.SIZE-1]])) # 3 Location states, 2 status states
        self.statespace_limits      = array(tile(_statespace_limits,(NUM_UAV,1)))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.motionNoise            = motionNoise # with some noise, when uav desires to transition to new state, remains where it is (loiter)
        self.LIMITS                 = tile(UAVAction.SIZE, (1,NUM_UAV))[0] # eg [3,3,3,3]
        self.isCommStatesCovered    = False # Don't have communications available yet, so no surveillance reward allowed.
        self.location_rect_vis      = None # 
        self.location_coord         = None
        self.uav_circ_vis           = None
        self.uav_text_vis           = None
        self.uav_sensor_vis         = None
        self.uav_actuator_vis       = None
        self.comms_line             = None
        self.dist_between_locations = self.RECT_GAP + self.LOCATION_WIDTH
#        self.SENSOR_REL_X = self.UAV_RADIUS - self.SENSOR_LENGTH
#        self.ACTUATOR_REL_Y = self.UAV_RADIUS - self.ACTUATOR_HEIGHT
#        state_space_dims = None # Number of dimensions of the state space
#        episodeCap = None       # The cap used to bound each episode (return to s0 after)

        super(PST,self).__init__(logger)
        if self.logger: self.logger.log("NUM_UAV:\t\t%d" % self.NUM_UAV)
        
    def resetLocalVariables(self):
        self.numCrashed = 0 # Number of crashed UAVs [n_c]
        
        
    def showDomain(self,s,a = 0):
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
        self.subplot_axes.set_xlim(0, 1 + UAVLocation.SIZE * self.dist_between_locations)
        self.subplot_axes.set_ylim(0, 1 + self.NUM_UAV)
        self.subplot_axes.xaxis.set_visible(False)
        self.subplot_axes.yaxis.set_visible(False)

        # Assign coordinates of each possible uav location on figure
        self.location_coord = [0.5 + (self.LOCATION_WIDTH / 2) + (self.dist_between_locations)*i for i in range(UAVLocation.SIZE-1)]
        crashLocationX = 1.0 + (self.dist_between_locations)*(UAVLocation.SIZE-1)
        self.location_coord.append(crashLocationX + self.LOCATION_WIDTH / 2)
        
         # Create rectangular patches at each of those locations
        self.location_rect_vis = [mpatches.Rectangle([0.5 + (self.dist_between_locations)*i, 0], self.LOCATION_WIDTH, self.NUM_UAV * 2, fc = 'w') for i in range(UAVLocation.SIZE-1)]
        self.location_rect_vis.append(mpatches.Rectangle([crashLocationX, 0], self.LOCATION_WIDTH, self.NUM_UAV * 2, fc = 'r'))
        [self.subplot_axes.add_patch(self.location_rect_vis[i]) for i in range(4)]
        self.comms_line = [lines.Line2D([0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*i, 0.5 + self.LOCATION_WIDTH + (self.dist_between_locations)*i + self.RECT_GAP],[self.NUM_UAV + 0.5, self.NUM_UAV + 0.5], linewidth = 3, color='black', visible=False) for i in range(UAVLocation.SIZE-2)]
        # Initialize list of circle objects
        
        uav_x = self.location_coord[UAVLocation.MAINTENANCE]
        
#            self.uav_vis_list = [UAVDispObject(uav_id) for uav_id in range(0,self.NUM_UAV)]
        # Update the member variables storing all the figure objects
        self.uav_circ_vis = [mpatches.Circle((uav_x,1+uav_id), self.UAV_RADIUS, fc="w") for uav_id in range(0,self.NUM_UAV)]
        self.uav_text_vis = [pl.text(0, 0, 0) for uav_id in range(0,self.NUM_UAV)]
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
        for uav_id in range(0,self.NUM_UAV):
            # Assign all the variables corresponding to this UAV for this iteration;
            # this could alternately be done with a UAV class whose objects keep track
            # of these variables.  Elect to use lists here since ultimately the state
            # must be a vector anyway.
            uav_ind = uav_id * UAVIndex.SIZE
            uav_location = s[uav_ind + UAVIndex.UAV_LOC] # State index corresponding to the location of this uav
            uav_fuel = s[uav_ind + UAVIndex.UAV_FUEL]
            uav_sensor = s[uav_ind + UAVIndex.UAV_SENS_STATUS]
            uav_actuator = s[uav_ind + UAVIndex.UAV_ACT_STATUS]
            
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
            
        if self.isCommStatesCovered == True: # We have comms coverage: draw a line between comms states to show this
            if self.numHealthySurveil > 0: # We also have UAVs in surveillance; color the comms line black
                commsColor = 'black'
            else: commsColor = 'red'
            [self.comms_line[i].set_color(commsColor) for i in range(len(self.comms_line))]
            [self.comms_line[i].set_visible(True) for i in range(len(self.comms_line))]
        else: # No comms coverage
            if self.numHealthySurveil > 0: # Surveillance but no comms; indicate with an X at the surveillance state
                self.comms_line[len(self.comms_line)-1].set_color('red')
                self.comms_line[len(self.comms_line)-1].set_visible(True)
                self.subplot_axes.add_line(lines.Line2D([self.location_coord[i] + self.LOCATION_WIDTH, self.location_coord[i] + self.LOCATION_WIDTH],[self.NUM_UAV + 0.75, self.NUM_UAV + 0.25], linewidth = 3, color='red', visible=True))
        [self.subplot_axes.add_line(self.comms_line[i]) for i in range(len(self.comms_line))] # Only visible lines actually appear
        pl.draw()
        sleep(2)
#===============================================================================
    def showLearning(self,representation):
        pass
    def step(self,s,a):
        ns = s.copy() # no concerns about affecting state mid-step here
        actionVector = id2vec(a,self.LIMITS) # returns list of form [0,1,0,2] corresponding to action of each uav
#DEBUG        print 'action vector selected:',actionVector
        
        # Create array with value 0 in all states corresponding to comms locations
        # Update this while iterate over each UAV; used to determine if comms link is functioning
        # i.e., all communications states are occupied by UAVs with functioning actuators and sensors
        self.isCommStatesCovered = False
        
        self.numHealthySurveil = 0
        self.fuelUnitsBurned = self.NUM_UAV
        distanceTraveled = 0
        
        totalStepReward = 0
        
        for uav_id in arange(0,self.NUM_UAV):            
            uav_ind = uav_id * UAVIndex.SIZE
            uav_location_ind = uav_ind + UAVIndex.UAV_LOC # State index corresponding to the location of this uav
            # First check for crashing uavs; short-circuit the loop
            # NOTE: Kemal said penalty only occurred once, but that doesn't match tutorial description + results.
#            if(s[uav_location_ind] == UAVLocation.CRASHED):
#                continue
            
            uav_fuel_ind = uav_ind + UAVIndex.UAV_FUEL
            uav_sensor_ind = uav_ind + UAVIndex.UAV_SENS_STATUS
            uav_actuator_ind = uav_ind + UAVIndex.UAV_ACT_STATUS
            
            uav_action = actionVector[uav_id]
            ##### STATE TRANSITIONS #####
            
            # Sensor failure transition
            if(random.random() < self.P_SENSOR_FAIL):
                ns[uav_sensor_ind] = SensorState.FAILED
#DEBUG                    print 'UAV',uav_id,'Failed sensor!'

            # Actuator failure transition
            if(random.random() < self.P_ACT_FAIL):
                ns[uav_actuator_ind] = ActuatorState.FAILED
                
            # Position state transition
            if(uav_action == UAVAction.ADVANCE):
                if(random.random() >= self.motionNoise): # With some noise, don't transition to new state
                    ns[uav_location_ind] += 1
                    distanceTraveled += 1
            elif(uav_action == UAVAction.RETREAT):
                if(random.random() >= self.motionNoise): # With some noise, don't transition to new state
                    ns[uav_location_ind] -= 1
                    distanceTraveled += 1
            
            # Fuel burn transition
            ns[uav_fuel_ind] -= self.NOM_FUEL_BURN
            
            # Take refuel/repair actions as necessary
            if(uav_action == UAVAction.LOITER):
                if(ns[uav_location_ind] == UAVLocation.REFUEL): # Refuel those which remain at base, immediately
                    ns[uav_fuel_ind] = self.FULL_FUEL
                    self.fuelUnitsBurned -= 1
                elif(ns[uav_location_ind] == UAVLocation.MAINTENANCE): # Repair those which remain in maint
                    ns[uav_sensor_ind] = SensorState.RUNNING
                    ns[uav_actuator_ind] = ActuatorState.RUNNING
                    ns[uav_fuel_ind] += self.NOM_FUEL_BURN # No fuel used when at base, remove fuel burned above
                    self.fuelUnitsBurned -= 1
            if (ns[uav_fuel_ind] < 1): # We just crashed a UAV! Failed to get back to base after action a
                self.numCrashed += 1
                ns[uav_fuel_ind] = 0 # Prevent negative numbers
#                break
            # Note above that we do not 'break', since the matlab version has reward for crashed UAVs also.  Makes sense if using s and not ns, as below.
            
            # Update status of comms availability / surveillance
            # TODO - matlab code uses previous location to determine if surveillance and comms are available; is this desired?
            if(s[uav_location_ind] == UAVLocation.COMMS):
                self.isCommStatesCovered = True
                # If actuator and sensor works, surveillance available in this state
                # If this uav is also in the surveillance region, then increment the number of uavs available for surveilance
            elif((s[uav_location_ind] == UAVLocation.SURVEIL) and (s[uav_sensor_ind] == SensorState.RUNNING)):
                  self.numHealthySurveil += 1
                      
        ##### Compute reward #####
        if(self.isCommStatesCovered == True):
            totalStepReward += self.SURVEIL_REWARD * min(self.NUM_TARGET, self.numHealthySurveil)
        if self.isTerminal(ns): totalStepReward += self.CRASH_REWARD
#        print 's, ns, surveilreward, crashreward, fuelburned', s, ns, self.SURVEIL_REWARD * min(self.NUM_TARGET, self.numHealthySurveil), self.isTerminal(ns), self.fuelUnitsBurned
        totalStepReward += self.FUEL_BURN_REWARD_COEFF * self.fuelUnitsBurned + self.MOVE_REWARD_COEFF * distanceTraveled # Presently movement penalty is set to 0
#debug        print totalStepReward,ns,self.isTerminal(ns)
        return totalStepReward,ns,self.isTerminal(ns)
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
    def s0(self):
        self.resetLocalVariables()
        returnList = zeros(self.NUM_UAV * UAVIndex.SIZE, dtype='int')
        for uav_ind in arange(0,self.NUM_UAV):
            returnList[uav_ind*UAVIndex.SIZE:(uav_ind+1)*UAVIndex.SIZE] = array([UAVLocation.MAINTENANCE, self.FULL_FUEL, ActuatorState.RUNNING, SensorState.RUNNING])
        return array(returnList) # Omits final index
    def possibleActions(self,s):
        # return the id of possible actions
        # find empty blocks (nothing on top)
        validActions = [] # Contains a list of uav_actions lists, e.g. [[0,1,2],[0,1],[1,2]] with the index corresponding to a uav.
        # First, enumerate the possible actions for each uav
        for uav_id in range(0,self.NUM_UAV):
            uav_actions = []
            uav_ind = uav_id * UAVIndex.SIZE
            uav_location = s[uav_ind + UAVIndex.UAV_LOC] # State index corresponding to the location of this uav
            uav_fuel = s[uav_ind + UAVIndex.UAV_FUEL]
#            uav_sensor = s[uav_ind + UAVIndex.UAV_SENS_STATUS]
            uav_actuator = s[uav_ind + UAVIndex.UAV_ACT_STATUS]
            
            if(uav_actuator == ActuatorState.RUNNING or \
            uav_location == UAVLocation.REFUEL or \
            uav_location == UAVLocation.MAINTENANCE):
                uav_actions.append(UAVAction.LOITER)
                
            if(uav_fuel > 0): # This UAV is not crashed
                if(uav_location != UAVLocation.SURVEIL and \
                uav_actuator == ActuatorState.RUNNING):
                    uav_actions.append(UAVAction.ADVANCE)
                    
                if(uav_location != UAVLocation.MAINTENANCE):
                    uav_actions.append(UAVAction.RETREAT)
                    
            # This UAV has no actions available to itgive it a dummy action for now
            if(len(uav_actions) < 1): uav_actions.append(UAVAction.LOITER)
                
            if(isinstance(uav_actions,int)):
                validActions.append([uav_actions])
            else:
                validActions.append(uav_actions)
#        print 's,a',s,validActions
#        print array(self.vecList2id(validActions, UAVAction.SIZE))
        return array(self.vecList2id(validActions, UAVAction.SIZE)) # TODO place this in tools
    
            # Given a list of lists 'validActions' of the form [[0,1,2],[0,1],[1,2],[0,1]]... return
        # unique id for each permutation between lists; eg above, would return 3*2*2*2 values
        # ranging from 0 to 3^4 -1 (3 is max value possible in each of the lists)
        # Takes parameters of totalActionSpaceSize, here 3^4-1, and number of actions available
        # to each agent, here 3
            
            ####### TODO place this in Tools
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
#                print partialActionAssignment,',,',limits
                actionIDs.append(vec2id(partialActionAssignment, limits)) # eg [0,1,0,2] and [3,3,3,3]
            else:
                self.vecList2idHelper(x,actionIDs,ind+1,partialActionAssignment, maxValue,limits) # TODO remove self
#        return actionIDs
    def isTerminal(self,s):
        if self.numCrashed > 0: return True
        else: return False
# \endcond
if __name__ == '__main__':
        random.seed(0)
        p = PST(NUM_UAV = 3, motionNoise = 0);
        p.test(1000)
        
        
        
    

