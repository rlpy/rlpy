######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
# This object encodes fixed policies for some of the domains.
# 1. InvertedPendulum
from Policy import *
class FixedPolicy(Policy):
    supportedDomains = ['Pendulum_InvertedBalance','BlocksWorld','IntruderMonitoring']
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
            
            #Random Action with 5 percent chance
            if random.rand() < .05 or domain.isTerminal(s):
                return randSet(domain.possibleActions(s))

            #non-Random Policy
            #next_block is the block that should be stacked on the top of the tower
            #wrong_block is the highest block stacked on the top of the next_block    
            blocks = domain.blocks
            next_block = 1
            for b in arange(1,blocks):
                if s[b] != b-1:
                    break
                else:
                    next_block = b+1
            # See if the next_block is empty
            onTop = domain.top(next_block,s)
            if not len(onTop):
                #next_block is clear
                action = [next_block, next_block-1]
            else:
                # next_block is not clear. Find the one on the top and put it on the table
                wrong_block = onTop[0]
                while True:
                    onTop = domain.top(wrong_block,s)
                    if len(onTop) == 0:
                        break
                action = [wrong_block, wrong_block] # Meaning put it on the table
            print 'STATE=', s
            print 'ACTION=', action
            return vec2id(action,[blocks, blocks])
        if className(self.representation.domain) == 'IntruderMonitoring':
            # Each UAV assign themselves to a target
            # Each UAV finds the closest danger zone to its target and go towards there. 
            # If UAVs_num > Target, the rest will hold position
            #Move all agents based on the taken action
            domain  = self.representation.domain
            agents  = array(s[:domain.NUMBER_OF_AGENTS*2].reshape(-1,2))
            targets = array(s[domain.NUMBER_OF_AGENTS*2:].reshape(-1,2))
            zones   = domain.danger_zone_locations
            actions = ones(len(agents))*4 # Default action is hold
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
            return vec2id(actions,ones(len(agents))*5)