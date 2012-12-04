import sys, os
import csv
import networkx as nx
import matplotlib.pyplot as plt
#Add all paths
sys.path.insert(0, os.path.abspath('.'))
from Tools import *

networkGraph = nx.Graph()
networkGraph.add_node(0, node_color = 'g')
networkGraph.add_node(1, node_color = 'r')
networkPos = nx.circular_layout(networkGraph)
nx.draw_networkx_nodes(networkGraph,networkPos,nodelist=[0,1],node_color="b")
#nx.draw_networkx_nodes(networkGraph,networkPos,nodelist=[1],node_color="r")

# self.networkGraph.add_edge(computer_id,neighbor_id, edge_color = "k")

plt.show()
