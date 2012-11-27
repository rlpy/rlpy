ALBORZ ACTION-

Blocksworld-
There's a minor typo; towersize is defined at the top of the class, but self.towerSize is referenced in all other places.




PST-

State transitions appear reasonable / sound
Note: UAV Crashes when fuel reaches 0 after an action (ie, with FULL_FUEL = 10, if the uav hasn't visited the base after the 10th action, it crashes. UAV cannot exist with 0 fuel outside of base)

Multiple chained communication states between base and goal are possible by setting NUM_COMMS_LOC

Direct Cost of fuel burn not mentioned in the tutorial, but included in paper; incorporate in model, currently FUEL_BURNED_REWARD_COEFF = 0.0

Visualization forthcoming





NetworkAdmin -

Soundness needs verification

Visualization will be changed; graph needs to be directed; presently nodes compete for 'good' or 'bad' edges, since the edge color is set by each node's output [i.e. if a good node and bad node are connected, the good node will attempt to set its outgoing edge green, and the bad node will attempt to set the same edge red]
