import sys, os

import csv
import networkx as nx
#



#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

########################################################
# Robert Klein, Alborz Geramifard Nov 26 2012 at MIT #
########################################################
# Persistent Search and Track Mission with:
# NUM_UAV:       num vehicles present
# NUM_COMMS_LOC  num intermediate communication states
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
########################################################

class UAVLocation:
    NUM_COMMS_LOC = 1 # Each comms location implicitly lies on [2, SIZE - 2]
    SIZE = 3 + NUM_COMMS_LOC   # Manually inputted
    # Possible location states, excluding communicatinons
    CRASHED = 0
    BASE_LOC = 1
    SURVEIL_LOC = SIZE-1
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

class PST(Domain):
    
    episodeCap = 100 # 100 used in tutorial
    
    FULL_FUEL = 10 # Number of fuel units at start
    P_NOM_FUEL_BURN = 0.8 # Probability of nominal (1 unit) fuel burn on timestep
 #   P_TWO_FUEL_BURN = 1.0 - P_NOM_FUEL_BURN # Probability of 2-unit fuel burn on timestep
    P_ACT_FAIL = 0.02 # Probability that actuators fail on this timestep
    P_SENSOR_FAIL = 0.05 #Probability that sensors fail on this timestep
    CRASH_REWARD_COEFF = -2.0 # Negative reward coefficient for running out of fuel (applied on every step) [C_crash]
    SURVEIL_REWARD_COEFF = 1.5 # Positive reward coefficient for performing surveillance on each step [C_cov]
    FUEL_BURN_REWARD_COEFF = 0.0 # Negative reward coefficient: for fuel burn penalty, not mentioned in MDP Tutorial
    numCrashed = 0 # Number of crashed UAVs [n_c]
    numHealthySurveil = 0 #Number of UAVs in surveillance area with working sensor and actuator [n_s]
    commsAvailable = 0 #1 if at least 1 UAV is in comms area with healthy sensor [I_comm]
    fuelUnitsBurned = 0
    LIMITS = []
    
    NUM_UAV = 0 # Number of UAVs present in the mission
    REFUEL_RATE = 1 # Rate of refueling per timestep
    NOM_FUEL_BURN = 1 # Nominal rate of fuel depletion selected with probability P_NOM_FUEL_BURN
    STOCH_FUEL_BURN = 2 # Alternative fuel burn rate
    

    
    ###
    
    def __init__(self, NUM_UAV = 3, motionNoise = 0):
        self.NUM_UAV                = NUM_UAV
        self.states_num             = NUM_UAV * UAVIndex.SIZE       # Number of states (UAV_LOC, UAV_FUEL...)
        self.actions_num            = pow(UAVAction.SIZE,NUM_UAV)    # Number of Actions: ADVANCE, RETREAT, LOITER
        _statespace_limits = vstack([[0,UAVLocation.SIZE-1],[0,self.FULL_FUEL-1],[0,ActuatorState.SIZE-1],[0,SensorState.SIZE-1]]) # 3 Location states, 2 status states
        self.statespace_limits      = tile(_statespace_limits,(NUM_UAV,1))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.motionNoise = motionNoise # with some noise, when uav desires to transition to new state, remains where it is (loiter)
        self.LIMITS                 = tile(UAVAction.SIZE, (1,NUM_UAV))[0] # eg [3,3,3,3]
#        state_space_dims = None # Number of dimensions of the state space
#        episodeCap = None       # The cap used to bound each episode (return to s0 after)
        super(PST,self).__init__()
    def showDomain(self,s,a = 0):
        pass
#===============================================================================
# #        if pass: # need to draw initial environment
# 
#        
#        
#        
#         #Draw the environment
#        if self.circles is None:
#           fig = pl.figure(1, (self.chainSize*2, 2))
#           ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
#           ax.set_xlim(0, self.chainSize*2)
#           ax.set_ylim(0, 2)
#           ax.add_patch(mpatches.Circle((1+2*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
#           ax.xaxis.set_visible(False)
#           ax.yaxis.set_visible(False)
#           self.circles = [mpatches.Circle((1+2*i, self.Y), self.RADIUS, fc="w") for i in range(self.chainSize)]
#           for i in range(self.chainSize):
#               ax.add_patch(self.circles[i])
#               if i != self.chainSize-1:
#                    fromAtoB(1+2*i+self.SHIFT,self.Y+self.SHIFT,1+2*(i+1)-self.SHIFT, self.Y+self.SHIFT)
#                    if i != self.chainSize-2: fromAtoB(1+2*(i+1)-self.SHIFT,self.Y-self.SHIFT,1+2*i+self.SHIFT, self.Y-self.SHIFT, 'r')
#               fromAtoB(.75,self.Y-1.5*self.SHIFT,.75,self.Y+1.5*self.SHIFT,'r',connectionstyle='arc3,rad=-1.2')
#               pl.show(block=False)
#            
#        [p.set_facecolor('w') for p in self.circles]
#        self.circles[s].set_facecolor('k')
#        pl.draw()
#===============================================================================
    def showLearning(self,representation):
        pass
    def step(self,s,a):
        ns = s # no concerns about affecting state mid-step here
        actionVector = id2vec(a,self.LIMITS) # returns list of form [0,1,0,2] corresponding to action of each uav
        print 'action vector selected:',actionVector
        
        # Create array with value 0 in all states corresponding to comms locations
        # Update this while iterate over each UAV; used to determine if comms link is functioning
        # i.e., all communications states are occupied by UAVs with functioning actuators and sensors
        commStatesCovered = [0] * UAVLocation.SIZE
        commStatesCovered[UAVLocation.BASE_LOC] = 1
        commStatesCovered[UAVLocation.CRASHED] = 1
        commStatesCovered[UAVLocation.SURVEIL_LOC] = 1
        
        self.numHealthySurveil = 0
        self.fuelUnitsBurned = 0
        
        for uav_id in range(0,self.NUM_UAV):            
            uav_ind = uav_id * UAVIndex.SIZE
            uav_location_ind = uav_ind + UAVIndex.UAV_LOC # State index corresponding to the location of this uav
            # First check for crashing uavs; short-circuit the loop
            if(s[uav_location_ind] == UAVLocation.CRASHED):
                continue
            
            uav_fuel_ind = uav_ind + UAVIndex.UAV_FUEL
            uav_sensor_ind = uav_ind + UAVIndex.UAV_SENS_STATUS
            uav_actuator_ind = uav_ind + UAVIndex.UAV_ACT_STATUS
            
            uav_action = actionVector[uav_id]
            ##### STATE TRANSITIONS #####            
            # Position state transition
            if(uav_action == UAVAction.ADVANCE):
                if(random.random() > self.motionNoise): # With some noise, don't transition to new state
                    ns[uav_location_ind] += 1
            elif(uav_action == UAVAction.RETREAT):
                if(random.random() > self.motionNoise): # With some noise, don't transition to new state
                    ns[uav_location_ind] -= 1
            # else, action is loiter, no motion taken.
            
            if (ns[uav_location_ind] != UAVLocation.BASE_LOC): # Not at base, failures can occur
                # Fuel burn transition
                if(random.random() < self.P_NOM_FUEL_BURN):
                    ns[uav_fuel_ind] -= self.NOM_FUEL_BURN
                    self.fuelUnitsBurned += self.NOM_FUEL_BURN
                else:
                    ns[uav_fuel_ind] -= self.STOCH_FUEL_BURN
                    self.fuelUnitsBurned += self.STOCH_FUEL_BURN
                
                # Sensor failure transition
                if(random.random() < self.P_SENSOR_FAIL):
                    ns[uav_sensor_ind] = SensorState.FAILED
                    print 'UAV',uav_id,'Failed sensor!'
               
                # Actuator failure transition
                if(random.random() < self.P_ACT_FAIL):
                    ns[uav_actuator_ind] = ActuatorState.FAILED
                    print '### UAV',uav_id,'Failed actuator! ###'
                
                if (ns[uav_fuel_ind] < 1): # We just crashed a UAV! Failed to get back to base after action a
                    self.numCrashed += 1
                    ns[uav_location_ind] = UAVLocation.CRASHED
                    print '########### UAV',uav_id,'has crashed! ############'
                    continue
                
                if(ns[uav_actuator_ind] == ActuatorState.RUNNING): # If actuator works, comms are available in this state
                    commStatesCovered[ns[uav_location_ind]] = 1
                    # If actuator and sensor works, surveillance available in this state
                    # If this uav is also in the surveillance region, then increment the number of uavs available for surveilance
                    if((ns[uav_location_ind] == UAVLocation.SURVEIL_LOC) and (ns[uav_sensor_ind] == SensorState.RUNNING)):
                          self.numHealthySurveil += 1
                
            else: # We transitioned to or loitered in Base
                ns[uav_fuel_ind] += self.REFUEL_RATE
                ns[uav_fuel_ind] = bound(ns[uav_fuel_ind], 0, self.FULL_FUEL)
                ns[uav_sensor_ind] = SensorState.RUNNING;
                ns[uav_actuator_ind] = ActuatorState.RUNNING;
                print 'UAV',uav_id,'at base.'
        
        ##### Compute reward #####
        totalStepReward = 0
        if(sum(commStatesCovered) == UAVLocation.SIZE): # All comms states have at least 1 functional UAV [I_comm = 1]
            totalStepReward += self.SURVEIL_REWARD_COEFF * self.numHealthySurveil
        totalStepReward += self.CRASH_REWARD_COEFF * self.numCrashed
        totalStepReward += self.FUEL_BURN_REWARD_COEFF * self.fuelUnitsBurned # Presently this component is set to zero.
 
        return totalStepReward,ns,self.NOT_TERMINATED
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
    def s0(self):
        returnList = []
        for dummy in range(0,self.NUM_UAV):
            returnList = returnList + [UAVLocation.BASE_LOC, self.FULL_FUEL, ActuatorState.RUNNING, SensorState.RUNNING]
        return returnList # Omits final index
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
            uav_sensor = s[uav_ind + UAVIndex.UAV_SENS_STATUS]
            uav_actuator = s[uav_ind + UAVIndex.UAV_ACT_STATUS]
            
            uav_actions.append(UAVAction.LOITER)
            
            if (uav_location == UAVLocation.CRASHED): # Only action available is loiter.
                pass
            elif (uav_location == UAVLocation.BASE_LOC):
                if(uav_fuel == self.FULL_FUEL): # Without full fuel, uav cannot leave base
                    uav_actions.append(UAVAction.ADVANCE)
            elif (uav_location == UAVLocation.SURVEIL_LOC):
                uav_actions.append(UAVAction.RETREAT)
            else: # uav_location is in one of the communications regions
                uav_actions.append(UAVAction.ADVANCE)
                uav_actions.append(UAVAction.RETREAT)
            if(isinstance(uav_actions,int)):
                validActions.append([uav_actions])
            else:
                validActions.append(uav_actions)
        print validActions
        return self.vecList2id(validActions, UAVAction.SIZE) # TODO place this in tools
    
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
        limits = tile(maxValue, (1,len(x)))[0] # eg [3,3,3,3] # TODO redundant computation
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
    
if __name__ == '__main__':
        random.seed(0)
        p = PST();
        p.test(100)
        
        
        
    