## TODO
# Rather than iterating through all nodes and neighbors each time want to display visual,
# Only update the graph for those nodes which have changed.
# Doesn't affect performance of domain itself though.

import sys, os

#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

##########################################################
# Robert H Klein, Alborz Geramifard Nov 21st 2012 at MIT #
##########################################################
# Download NetworkX package for visualization.
# Ubuntu: sudo apt-get install python-networkx
#
# Network Administrator with n computers, at most 1 reboot
# action allowed per timestep.
# Penalty -1.75 for taking a reboot action. [netting -0.75
# since the computer status becomes 'RUNNING' after action].
# State is the vector of binary computer statuses:
# RUNNING = 1 for working computers, BROKEN = 0 otherwise.
# Example: [1 1 0 1] -> computers 0,1,3 are RUNNING,
# computer 2 is BROKEN.
########################################################

########################################################
##########               INPUT                ##########
# Each row is implicitly indexed starting at 0,
# corresponding to the id of a computer.
# The sequence of numbers (arbitrary order) corresponds
# to the computers connected to this one.
# NOTE: The graph is assumed to be bidirectional.
# You dont have to specify both edges between the nodes!
# 1,2 on the first line means these edges: (0,1),(1,0),(2,0),(0,2)
# Each line has to have at least one element
## @author: Robert H Klein and Alborz Geramifard
class SystemAdministrator(Domain):

    NEIGHBORS = []          # Each cell corresponds to a computer; contents of cell is a list of neighbors connected to that computer
    UNIQUE_EDGES = []       # A list of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
    
    P_SELF_REPAIR = 0.04
    P_REBOOT_REPAIR = 1.0
    
    REBOOT_REWARD = -1.75
    # Computer "up" reward implicitly 1; tune other rewards relative to this.   
     
    episodeCap = 200        # 200 used in tutorial
    gamma      = .95        # Based on IJCAI01 Paper
    # Plotting Variables
    networkGraph = None     # Graph of network used for visualization
    networkPos = None       # Position of network graph
    
    # Possible values for each computer
    BROKEN, RUNNING = 0,1
    _NUM_VALUES = 2         # Number of values possible for each state, must be hand-coded to match number defined above
            
    ## Note that you must pass a network map name as well as its format type.
    # @see SystemAdministrator(Domain)
    def __init__(self, networkmapname='/Domains/SystemAdministratorMaps/20MachTutorial.txt', logger = None):
        path                    = os.getcwd() + networkmapname
        self.loadNetwork(path)   
        # TODO Need a check here for degenerate
        self.states_num             = len(self.NEIGHBORS)       # Number of states
        self.actions_num            = self.states_num + 1     # Number of Actions, including no-op
        self.statespace_limits      = tile([0,self._NUM_VALUES-1],(self.states_num,1))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        super(SystemAdministrator,self).__init__(logger)
        if self.logger: 
            self.logger.log('Edges:\t\t%s' % str(self.UNIQUE_EDGES))
            self.logger.log('Neighbors:\t%s' % str(self.NEIGHBORS))
#        for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,self.s0())):
#            [self.logger.log("Node:\t%d\t Neighbors:\t%d" % self.NEIGHBORS[i]) for i in self.NEIGHBORS]

    ## @param path: Path to the map file, of form '/Domains/SystemAdministratorMaps/<mapname>.txt'
    # @param maptype: Specify the format for the map file, 'eachNeighbor' or 'edges'.
    # @param numNodes: Number of nodes in the map.
    # @see: SystemAdministrator(Domain) for a description of 'eachNeighbor' and 'edges'.
    #
    # @return: the tuple (_Neighbors, _Edges), where each cell of _Neighbors is a list
    # containing the neighbors of computer node <i> at index <i>, and _Edges is a list
    # of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
    def loadNetwork(self, path):
        _Neighbors = []
        f = open(path, 'rb')
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            _Neighbors.append(map(int,row))
        self.computers_num = len(_Neighbors)
        self.setUniqueEdges(_Neighbors)
        self.setNeighbors()
    def showDomain(self,s,a = 0):
        if self.networkGraph is None: #or self.networkPos is None:
            self.networkGraph = nx.Graph()
            # enumerate all computer_ids, simulatenously iterating through neighbors list and compstatus
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
                self.networkGraph.add_node(computer_id, node_color = "w") # Add a node to network for each computer
            for uniqueEdge in self.UNIQUE_EDGES:
                    self.networkGraph.add_edge(uniqueEdge[0],uniqueEdge[1], edge_color = "k") # Add an edge between each neighbor
            self.networkPos = nx.circular_layout(self.networkGraph)
            nx.draw_networkx_nodes(self.networkGraph, self.networkPos, node_color="w")
            nx.draw_networkx_edges(self.networkGraph, self.networkPos, edges_color="k")
            nx.draw_networkx_labels(self.networkGraph, self.networkPos)
            pl.show()
        else:
            pl.clf()
            blackEdges = []
            redEdges = []
            greenNodes = []
            redNodes = []
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
#DEBUG                print "compid, neighbors, compstatus",computer_id, neighbors, compstatus
                if(compstatus == self.RUNNING):
                    greenNodes.append(computer_id)
                else:
                    redNodes.append(computer_id)
            for uniqueEdge in self.UNIQUE_EDGES: # Iterate through all unique edges
                if(s[uniqueEdge[0]] == self.RUNNING and s[uniqueEdge[1]] == self.RUNNING):
                    # Then both computers are working
                    blackEdges.append(uniqueEdge)
                else: # If either computer is BROKEN, make the edge red
                    redEdges.append(uniqueEdge)
#DEBUG            print "blackEdges",blackEdges
#DEBUG            print "gnnodes",greenNodes
#DEBUG            print "rednodes",redNodes
#DEBUG            print "rededges",redEdges
            # "if redNodes", etc. - only draw things in the network if these lists aren't empty / null
            if redNodes:    nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=redNodes, node_color="r",linewidths=2)
            if greenNodes:  nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=greenNodes, node_color="w",linewidths=2)
            if blackEdges:  nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=blackEdges, edge_color="k",width=2,style='solid')
            if redEdges:    nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=redEdges, edge_color="k",width=2,style='dotted')
        nx.draw_networkx_labels(self.networkGraph, self.networkPos)
        pl.draw()
    def step(self,s,a):
        #ns = s[:] # make copy of state so as not to affect original mid-step
        ns = s.copy()
        totalRebootReward = 0
        for computer_id, compstatus in enumerate(s):
            if(a == computer_id): #Reboot action on this computer
                totalRebootReward += self.REBOOT_REWARD
                # NOTE can break up if-statement below to separate cases
                if (random.random() <= self.P_REBOOT_REPAIR):
                    ns[computer_id] = self.RUNNING
                else:
                    ns[computer_id] = self.BROKEN
            else: # Transition to new state probabilistically
                if (compstatus == self.RUNNING):
                    # take neighbors of computer_id and sum over each of their current values
                    sumOfNeighbors = sum([s[i] for i in self.NEIGHBORS[computer_id]])
                    # TODO this expression should be a function, or something
                    p_broken = 1.0 - (0.45 + 0.5 * (1+sumOfNeighbors) / (1+len(self.NEIGHBORS[computer_id])))
#                    print "P_broken",p_broken
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
        #return [self.RUNNING for dummy in arange(0,self.state_space_dims)] # Omits final index
        return array([self.RUNNING]* self.state_space_dims)
    ## @param neighborsList: each element at index <i> is a list of nodes connected to the node at <i>.
    # @return: a list of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
    def setUniqueEdges(self, neighborsList):
        # set Unique Edges of the network (all edges are bidirectional)
        # the lowest computer_id [eg, edges (0,3) and (3,0) discard (3,0)]
        self.UNIQUE_EDGES = []
        for computer_id, neighbors in enumerate(neighborsList):
            for neighbor_id in neighbors:
                edge = (min(neighbor_id,computer_id), max(neighbor_id,computer_id))
                found = [t for t in self.UNIQUE_EDGES if t[0] == edge[0] and t[1] == edge[1]]
                if found == []:
                    self.UNIQUE_EDGES.append(edge)
    def setNeighbors(self):
        self.NEIGHBORS = {} # Initialize list so we don't get out of bounds errors
        for edgePair in self.UNIQUE_EDGES:
            # Add each node as a neighbor to each other
            s,d = edgePair
            if s in self.NEIGHBORS:
                self.NEIGHBORS[s] += [d]
            else:
                self.NEIGHBORS[s] = [d]
            if d in self.NEIGHBORS:
                self.NEIGHBORS[d] += [s]
            else:
                self.NEIGHBORS[d] = [s]
        for i in range(self.computers_num):
            self.NEIGHBORS[i] = array(self.NEIGHBORS[i])
if __name__ == '__main__':
        random.seed(0)
        p = SystemAdministrator(networkmapname='/SystemAdministratorMaps/8Ring.txt');
        #p = SystemAdministrator(networkmapname='/SystemAdministratorMaps/5Machines.txt');
        #p = SystemAdministrator(networkmapname='/SystemAdministratorMaps/10Machines.txt');
        #p = SystemAdministrator(networkmapname='/SystemAdministratorMaps/20MachTutorial.txt');
        p.test(1000)
     