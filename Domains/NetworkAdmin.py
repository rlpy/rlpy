import sys, os
import csv
import networkx as nx
import matplotlib.pyplot as plt
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

########################################################
# Robert Klein, Alborz Geramifard Nov 21st 2012 at MIT #
########################################################
# Download NetworkX package for visualization.
# Ubuntu: sudo apt-get install python-networkx
########################################################


class NetworkAdmin(Domain):
    NEIGHBORS = [] # Tuple of neighbors to this computer ['c' in tutorial]
    
    P_SELF_REPAIR = 0.04
    P_REBOOT_REPAIR = 1.0
    
    REBOOT_REWARD = -0.75
    # Computer "up" reward implicitly 1; tune other rewards relative to this.   
     
    episodeCap = 100 # TODO this seems rather arbitrary

    networkGraph = None #Graph of network used for visualization
    networkPos = None
    
    # Possible values for each computer
    BROKEN, RUNNING = 0,1
    _NUM_VALUES = 2 # Number of values possible for each state, must be hand-coded to match number defined above
    
    
    def getNetworkMap(self, path):
        _Neighbors = []
        with open(path, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                _Neighbors.append(map(int,row))
        return _Neighbors
    
    def __init__(self, networkmapname='/NetworkAdminMaps/5Machines.txt'):
        path                    = os.getcwd() + networkmapname
        self.NEIGHBORS          = self.getNetworkMap(path)
        # TODO Need a check here for degenerate
        
        self.states_num             = len(self.NEIGHBORS)       # Number of states
        self.actions_num            = self.states_num + 1     # Number of Actions, including no-op
        self.statespace_limits      = tile([0,self._NUM_VALUES-1],(self.states_num,1))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
#        state_space_dims = None # Number of dimensions of the state space
#        episodeCap = None       # The cap used to bound each episode (return to s0 after)
        super(NetworkAdmin,self).__init__()
    
    def showDomain(self,s,a = 0):
        if self.networkGraph is None: #or self.networkPos is None:
            self.networkGraph = nx.Graph()
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
                self.networkGraph.add_node(computer_id, node_color = "w")
                for neighbor_id in neighbors:
                    self.networkGraph.add_edge(computer_id,neighbor_id, edge_color = "k")
            self.networkPos = nx.circular_layout(self.networkGraph)
            nx.draw_networkx_nodes(self.networkGraph, self.networkPos, node_color="w")
            nx.draw_networkx_edges(self.networkGraph, self.networkPos, edges_color="k")
            pl.show(block=False)
        else:
            blackEdges = []
            redEdges = []
            greenNodes = []
            redNodes = []
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
                print "compid, neighbors, compstatus",computer_id, neighbors, compstatus
                if(compstatus == self.RUNNING):
                    greenNodes.append(computer_id)
                    for neighbor_id in neighbors: blackEdges.append((computer_id, neighbor_id))
                else:
                    redNodes.append(computer_id)
                    for neighbor_id in neighbors: redEdges.append((computer_id, neighbor_id))
            print "blackEdges",blackEdges
            print "gnnodes",greenNodes
            print "rednodes",redNodes
            print "rededges",redEdges
            if redNodes:    nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=redNodes, node_color="r")
            if greenNodes:  nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=greenNodes, node_color="w")
            if blackEdges:  nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=blackEdges, edge_color="k")
            if redEdges:    nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=redEdges, edge_color="r")
        plt.draw()    
         
    
    def showLearning(self,representation):
        pass
    
    def step(self,s,a):
        ns = s # make copy of state so as not to affect original mid-step
        totalRebootReward = 0
        for computer_id, compstatus in enumerate(s):
            if(a == computer_id): #Reboot action on this computer
                totalRebootReward += self.REBOOT_REWARD
                # NOTE can break up if-statement below to separate cases
                if (random.random() < self.P_REBOOT_REPAIR):
                    ns[computer_id] = self.RUNNING
                else:
                    ns[computer_id] = self.BROKEN
            else: # Transition to new state probabilistically
                if (compstatus == self.RUNNING):
                    # take neighbors of computer_id and sum over each of their current values
                    sumOfNeighbors = sum([s[i] for i in self.NEIGHBORS[computer_id]])
                    # TODO this expression should be a function, or something
                    p_broken = 0.45 + 0.5 * (1+sumOfNeighbors) / (1+len(self.NEIGHBORS))
                    print "P_broken",p_broken
                    if(random.random() < p_broken ):
                        ns[computer_id] = self.BROKEN
 # Optional                    else: ns[computer_id] = self.RUNNING
                else:
                    if(random.random() < self.P_SELF_REPAIR):
                        ns[computer_id] = self.RUNNING
 # Optional                     else ns[computer_id] = self.BROKEN
        return sum(ns)+totalRebootReward,ns,self.NOT_TERMINATED
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
    
    def s0(self):
        return [self.RUNNING for dummy in range(0,self.state_space_dims)] # Omits final index
    
    def possibleActions(self,s):
    # Returns the list of possible actions in each state the vanilla version returns all of the actions
        return arange(self.actions_num)
    
if __name__ == '__main__':
        #p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
        random.seed(0)
        p = NetworkAdmin('/NetworkAdminMaps/5Machines.txt');
        p.test(1000)
     