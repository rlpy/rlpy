"""Fixed policy. Encodes fixed policies for particular domains."""

from Policy import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class FixedPolicy(Policy):

    policyName  = '' # The name of the desired policy, where applicable. Otherwise ignored.
    tableOfValues = None

    gridWorldPolicyNames = ['cw_circle', 'ccw_circle']

    def __init__(self, representation, logger, policyName = 'MISSINGNO', tableOfValues=None):
        self.policyName = policyName
        self.tableOfValues = tableOfValues
        super(FixedPolicy, self).__init__(representation,logger)

    supportedDomains = ['InfCartPoleBalance','BlocksWorld','IntruderMonitoring',\
                        'SystemAdministrator','MountainCar','PST','GridWorld',\
                        ]
    def pi(self,s, terminal, p_actions):
        if self.tableOfValues:
            return self.tableOfValues[(s)]
        return self.pi2(s)

    def pi2(self,s, terminal, p_actions):
        domain = self.representation.domain
        if not className(domain) in self.supportedDomains:
            print "ERROR: There is no fixed policy defined for %s" % className(domain)
            return None

        if className(domain) == 'GridWorld':
            # Actions are Up, Down, Left, Right
            if not self.policyName in self.gridWorldPolicyNames:
                print "Error: There is no GridWorld policy with name %s" % self.policyName
                return None

            if self.policyName == 'cw_circle':
                # Cycle through actions, starting with 0, causing agent to go in loop
                if not hasattr(self, "curAction"):
                    self.curAction = 0  # it doesn't exist yet, so initialize it [immediately incremented]
                while (not(self.curAction in domain.possibleActions(s))):
                    # We can't do something simple because of the order in which actions are defined
                    # must do switch statement
                    if self.curAction == 0: #up
                        self.curAction = 3
                    elif self.curAction == 3: #right
                        self.curAction = 1
                    elif self.curAction == 1: #down
                        self.curAction = 2
                    elif self.curAction == 2: # left
                        self.curAction = 0
                    else: print 'Something terrible happened...got an invalid action on GridWorld Fixed Policy'
    #                 self.curAction = self.curAction % domain.actions_num
            elif self.policyName == 'ccw_circle':
                # Cycle through actions, starting with 0, causing agent to go in loop
                if not hasattr(self, "curAction"):
                    self.curAction = 1  # it doesn't exist yet, so initialize it
                while (not(self.curAction in domain.possibleActions(s))):
                    # We can't do something simple because of the order in which actions are defined
                    # must do switch statement
                    if self.curAction == 3: #right
                        self.curAction = 0
                    elif self.curAction == 0: #up
                        self.curAction = 2
                    elif self.curAction == 2: #left
                        self.curAction = 1
                    elif self.curAction == 1: # down
                        self.curAction = 3
                    else: print 'Something terrible happened...got an invalid action on GridWorld Fixed Policy'
    #                 self.curAction = self.curAction % domain.actions_num

            else:
                print "Error: No policy defined with name %s, but listed in gridWorldPolicyNames" % self.policyName
                print "You need to create a switch statement for the policy name above, or remove it from gridWorldPolicyNames"
                return None
            return self.curAction

#             # Cycle through actions, starting with 0, causing agent to go in other direction
#             if not hasattr(pi, "curAction"):
#                 pi.curAction = domain.actions_num-1  # it doesn't exist yet, so initialize it
#             if not(pi.curAction in domain.possibleActions(s)):
#                 pi.curAction -= 1
#                 if pi.curAction < 0: pi.curAction = domain.actions_num-1




        if className(domain) == 'InfCartPoleBalance':
            # Fixed policy rotate the pendulum in the opposite direction of the thetadot
            theta, thetadot = s
            if thetadot > 0:
                return 2
            else:
                return 0
        if className(domain) == 'BlocksWorld':
            # Fixed policy rotate the blocksworld = Optimal Policy (Always pick the next piece of the tower and move it to the tower
            # Policy: Identify the top of the tower.
            # move the next piece on the tower with 95% chance 5% take a random action

            #Random Action with some probability
            #TODO fix isTerminal use here
            if random.rand() < .3 or domain.isTerminal():
                return randSet(domain.possibleActions(s))

            #non-Random Policy
            #next_block is the block that should be stacked on the top of the tower
            #wrong_block is the highest block stacked on the top of the next_block
            #Wrong_tower_block is the highest stacked on the top of the tower
            blocks = domain.blocks
            correct_tower_size = 0 # Length of the tower assumed to be built correctly.
            while True:
                # Check the next block
                block = correct_tower_size
                if (block == 0 and domain.on_table(block,s)) or domain.on(block,block-1,s):
                    #This block is on the right position, check the next block
                    correct_tower_size += 1
                else:
                    #print s
                    #print "Incorrect block:", block
                    # The block is on the wrong place.
                    # 1. Check if the tower is empty => If not take one block from the tower and put it on the table
                    # 2. check to see if this wrong block is empty => If not put one block from its stack and put on the table
                    # 3. Otherwise move this block on the tower

                    ###################
                    #1
                    ###################
                    if block != 0: # If the first block is in the wrong place, then the tower top which is table is empty by definition
                        ideal_tower_top     = block - 1
                        tower_top = domain.towerTop(ideal_tower_top,s)
                        if tower_top != ideal_tower_top:
                            # There is a wrong block there hence we should put it on the table first
                            return domain.getActionPutAonTable(tower_top) #put the top of the tower on the table since it is not correct
                    ###################
                    #2
                    ###################
                    block_top = domain.towerTop(block,s)
                    if block_top != block:
                        # The target block to be stacked is not empty
                        return domain.getActionPutAonTable(block_top)
                    ###################
                    #3
                    ###################
                    if block == 0:
                        return domain.getActionPutAonTable(block)
                    else:
                        return domain.getActionPutAonB(block,block-1)
        if className(domain) == 'IntruderMonitoring':
            # Each UAV assign themselves to a target
            # Each UAV finds the closest danger zone to its target and go towards there.
            # If UAVs_num > Target, the rest will hold position
            #Move all agents based on the taken action
            agents  = array(s[:domain.NUMBER_OF_AGENTS*2].reshape(-1,2))
            targets = array(s[domain.NUMBER_OF_AGENTS*2:].reshape(-1,2))
            zones   = domain.danger_zone_locations
            actions = ones(len(agents),dtype=integer)*4 # Default action is hold
            planned_agents_num = min(len(agents),len(targets))
            for i in arange(planned_agents_num):
                #Find cloasest zone (manhattan) to the corresponding target
                target          = targets[i,:]
                distances       = sum(abs(tile(target,(len(zones),1)) - zones),axis=1)
                z_row,z_col     = zones[argmin(distances),:]
                # find the valid action
                a_row,a_col     = agents[i,:]
                a = 4 # hold as a default action
                if a_row > z_row:
                    a = 0 # up
                if a_row < z_row:
                    a = 1 # down
                if a_col > z_col:
                    a = 2 # left
                if a_col < z_col:
                    a = 3 # right
                actions[i] = a
#                print "Agent=", agents[i,:]
#                print "Target", target
#                print "Zone", zones[argmin(distances),:]
#                print "Action", a
#                print '============'
            return vec2id(actions,ones(len(agents),dtype=integer)*5)
        if className(domain) == 'SystemAdministrator':
            # Select a broken computer and reset it
            brokenComputers = where(s==0)[0]
            if len(brokenComputers):
                return randSet(brokenComputers)
            else:
                return domain.computers_num
        if className(domain) == 'MountainCar':
            # Accelerate in the direction of the valley
            # WORK IN PROGRESS
            x,xdot = s
            if xdot > 0:
                return 2
            else:
                return 0
        if className(domain) == 'PST':
            # One stays at comm, n-1 stay at target area. Whenever fuel is lower than reaching the base the move back
            print s
            s       = domain.state2Struct(s)
            uavs    = domain.NUM_UAV
            print s
            return vec2id(zeros(uavs),ones(uavs)*3)
