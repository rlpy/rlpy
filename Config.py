global TEXPATH, CONDOR_CLUSTER_PREFIX
TEXPATH = r'/usr/texbin'         # Location of Latex bin files. "r" is used to indicate raw string avoiding escape character interpretation
CONDOR_CLUSTER_PREFIX = '/data' # not used anywhere as part of path, only as unique identifier distinguishing cluster from normal local machine. See isOnCluster()
 