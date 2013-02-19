global TEXPATH, CONDOR_CLUSTER_PREFIX
TEXPATH = '/usr/texbin'         # Location of Latex bin files
CONDOR_CLUSTER_PREFIX = '/data' # not used anywhere as part of path, only as unique identifier distinguishing cluster from normal local machine. See isOnCluster()
