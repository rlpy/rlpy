#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# TODO
# Rather than iterating through all nodes and neighbors each time want to display visual,
# Only update the graph for those nodes which have changed.
# Doesn't affect performance of domain itself though.

import sys, os

#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *

##########################################################
# \author Robert H Klein, Alborz Geramifard Nov 21st 2012 at MIT
##########################################################
# Download NetworkX package for visualization. \n
# Ubuntu: sudo apt-get install python-networkx \n
#
# Network Administrator with n computers, at most 1 reboot
# action allowed per timestep. \n
# Penalty -0.75 for taking a reboot action. \n
# State is the vector of binary computer statuses: \n
# RUNNING = 1 for working computers, BROKEN = 0 otherwise. \n
# Example: 
# [1 1 0 1] -> computers 0,1,3 are RUNNING,
# computer 2 is BROKEN.
#
# In visualization, broken computers are colored red,
# and any links to other computers change from solid to
# dotted, reflecting the higher probability of failure
# of those machines. \n \n
# -------------------INPUT-------------------\n
# Each row is implicitly indexed starting at 0,
# corresponding to the id of a computer. \n
# The sequence of numbers (arbitrary order) corresponds
# to the computers connected to this one. \n
# NOTE: The graph is assumed to be bidirectional.
# You dont have to specify both edges between the nodes! \n
# 1,2 on the first line means these edges: (0,1),(1,0),(2,0),(0,2). \n
# Each line has to have at least one element
class SystemAdministrator(Domain):

    NEIGHBORS       = [] # Each cell corresponds to a computer; contents of cell is a list of neighbors connected to that computer
    UNIQUE_EDGES    = [] # A list of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
    
    P_SELF_REPAIR   = 0.04
    P_REBOOT_REPAIR = 1.0
    
    IS_RING         = False # For ring structures, Parr enforces assymetry by having one machine get extra reward for being up.
    
    REBOOT_REWARD   = -0.75
    # Computer "up" reward implicitly 1; tune other rewards relative to this.   
     
    episodeCap      = 200        # 200 used in tutorial
    gamma           = .95        # Based on IJCAI01 Paper
    
    # Plotting Variables
    networkGraph    = None     # Graph of network used for visualization
    networkPos  = None       # Position of network graph
    
    # Possible values for each computer
    BROKEN, RUNNING = 0,1
    _NUM_VALUES = 2         # Number of values possible for each state, must be hand-coded to match number defined above
            
    # Note that you must pass a network map name as well as its format type.
    # @see SystemAdministrator(Domain)
    def __init__(self, networkmapname='/Domains/SystemAdministratorMaps/20MachTutorial.txt', logger = None):
        path                    = networkmapname
        self.IS_RING            = "ring.txt" in networkmapname.lower()
        self.loadNetwork(path)   
        # TODO Need a check here for degenerate
        self.actions_num            = self.computers_num + 1     # Number of Actions, including no-op
        self.statespace_limits      = tile([0,self._NUM_VALUES-1],(self.computers_num,1))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        super(SystemAdministrator,self).__init__(logger)
        if self.logger: 
            self.logger.log('Computers:\t%d' % self.computers_num)
            self.logger.log('Edges:\t\t%s' % str(self.UNIQUE_EDGES))
            self.logger.log('Neighbors:')
            for i in range(self.computers_num):
                self.logger.log('%d : %s' % (i,str(list(self.NEIGHBORS[i]))))
#        for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,self.s0())):
#            [self.logger.log("Node:\t%d\t Neighbors:\t%d" % self.NEIGHBORS[i]) for i in self.NEIGHBORS]

    # @param path: Path to the map file, of form '/Domains/SystemAdministratorMaps/<mapname>.txt'             [Elliott Note: Doxygen did not like this documentation because it listed]
    # @param maptype: Specify the format for the map file, 'eachNeighbor' or 'edges'.                         [@params that are not called, I did not fixed documentation used below. ]
    # @param numNodes: Number of nodes in the map.
    # @see: SystemAdministrator(Domain) for a description of 'eachNeighbor' and 'edges'.
    #
    # @return: the tuple (_Neighbors, _Edges), where each cell of _Neighbors is a list
    # containing the neighbors of computer node <i> at index <i>, and _Edges is a list
    # of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
	
	# @param path: Path to the map file, of form '/Domains/SystemAdministratorMaps/mapname.txt'
	# @return: the tuple (_Neighbors, _Edges), where each cell of _Neighbors is a list
    # containing the neighbors of computer node i at index i, and _Edges is a list
    # of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.
    def loadNetwork(self, path):
        _Neighbors = []
        f = open(path, 'rb')
        reader = csv.reader(f, delimiter=',')
        self.computers_num = 0
        for row in reader:
            row = map(int,row)
            _Neighbors.append(row)
            self.computers_num = max(max(row)+1,self.computers_num)
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
       # print 'action selected',a,s
        totalRebootReward = 0
        for computer_id, compstatus in enumerate(s):
            if(a == computer_id): #Reboot action on this computer
                totalRebootReward += self.REBOOT_REWARD
                # NOTE can break up if-statement below to separate cases
                if (random.random() <= self.P_REBOOT_REPAIR):
#                    print 'repaired comp',computer_id
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
        if (self.IS_RING and s[0] == self.RUNNING): totalRebootReward += 1 # Per Guestrin, Koller, Parr 2003, rings have enforced asymmetry on one machine
#        print s,ns,sum(s)+totalRebootReward
        return sum(s)+totalRebootReward,ns,self.NOT_TERMINATED
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
    def s0(self):
        return array([self.RUNNING for dummy in arange(0,self.state_space_dims)]) # Omits final index
#        return array([self.BROKEN]* self.state_space_dims)
        #arrTmp = array([self.BROKEN]* self.state_space_dims)
        #arrTmp[arange(10)] = self.RUNNING
        #return arrTmp
    def possibleActions(self,s):
        possibleActs = [computer_id for computer_id,compstatus in enumerate(s) if compstatus == self.BROKEN]
        possibleActs.append(self.computers_num) # append the no-op action
        return array(possibleActs)
        
    # @param neighborsList: each element at index i is a list of nodes connected to the node at i.
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
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/10Ring.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/20Ring.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/9Star.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/5Machines.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/5MachinesEdges.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/10Machines.txt');
        p = SystemAdministrator(networkmapname='SystemAdministratorMaps/16-5Branches.txt');
        #p = SystemAdministrator(networkmapname='SystemAdministratorMaps/20MachTutorial.txt');
        p.test(1000)
     