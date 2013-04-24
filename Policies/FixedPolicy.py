#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Bob Klein
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
# This object encodes fixed policies for some of the domains.
# 1. InvertedPendulum
from Policy import *
class FixedPolicy(Policy):
    supportedDomains = ['Pendulum_InvertedBalance','BlocksWorld','IntruderMonitoring','SystemAdministrator','MountainCar','PST']
    def pi(self,s):
        if not className(self.representation.domain) in self.supportedDomains:
            print "ERROR: There is no fixed policy defined for %s" % className(self.representation.domain)
            return None
        if className(self.representation.domain) == 'Pendulum_InvertedBalance':
            # Fixed policy rotate the pendulum in the opposite direction of the thetadot
            theta, thetadot = s
            if thetadot > 0:
                return 2
            else:
                return 0
        if className(self.representation.domain) == 'BlocksWorld':
            # Fixed policy rotate the blocksworld = Optimal Policy (Always pick the next piece of the tower and move it to the tower
            # Policy: Identify the top of the tower.
            # move the next piece on the tower with 95% chance 5% take a random action
            domain = self.representation.domain
            
            #Random Action with some probability
            if random.rand() < .3 or domain.isTerminal(s):
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
        if className(self.representation.domain) == 'IntruderMonitoring':
            # Each UAV assign themselves to a target
            # Each UAV finds the closest danger zone to its target and go towards there. 
            # If UAVs_num > Target, the rest will hold position
            #Move all agents based on the taken action
            domain  = self.representation.domain
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
        if className(self.representation.domain) == 'SystemAdministrator':
            # Select a broken computer and reset it
            brokenComputers = where(s==0)[0]
            if len(brokenComputers):
                return randSet(brokenComputers)
            else:
                return self.representation.domain.computers_num
        if className(self.representation.domain) == 'MountainCar':
            # Accelerate in the direction of the valley
            # WORK IN PROGRESS
            x,xdot = s
            if xdot > 0:
                return 2
            else:
                return 0
        if className(self.representation.domain) == 'PST':
            # One stays at comm, n-1 stay at target area. Whenever fuel is lower than reaching the base the move back
            domain = self.representation.domain
            print s
            s       = domain.state2Struct(s)
            uavs    = domain.NUM_UAV
            print s
            return vec2id(zeros(uavs),ones(uavs)*3)
