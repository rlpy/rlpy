"""This is specifically for use with the SCORE algorithm implemented in C++,
and is coupled to the EDMHelper_score.py module which interfaces with it."""

from Agent import Agent
from Tools import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__  = "Robert H. Klein"


class ClusterManager(Agent):

    is_incremental      = False

    use_sparse          = 0         # Use sparse operators for building A?
    lspi_iterations     = 0         # Number of LSPI iterations
    max_window          = 0         # Number of samples to be used to calculate the A and b matrices
    steps_between_LSPI  = 0         # Number of samples between each LSPI run.
    eps_steps       = 0         # Number of samples gathered so far
    epsilon             = 0         # Minimum l_2 change required to continue iterations in LSPI

    return_best_policy  = 0        # If this flag is activated, on each iteration of LSPI, the policy is checked with one simulation and in the end the theta w.r.t the best policy is returned. Note that this will require more samples than the intial set of samples provided to LSPI
    best_performance    = -inf     # In the "return_best_policy", The best perofrmance check is stored here through LSPI iterations
    best_theta          = None     # In the "return_best_policy", The best theta is stored here through LSPI iterations
    best_TD_errors      = None     # In the "return_best_policy", The TD_Error corresponding to the best theta is stored here through LSPI iterations
    extra_samples       = 0        # The number of extra samples used due to extra simulations

    prev_min_objective  = inf      # The previous minimum objective; used to check if we
                # will perform better after representation expansion or not; if not, halt.


    transitionStructList = []
    edmHelper       =       None # Helper object, which wraps edm and provides methods

    MAX_CLUSTERS        = 50        # Expected number of clusters, buffering the vector which stores them.


    #Representation Expansion
    re_iterations   = 0 # Maximum number of iterations over LSPI and Representation expansion
    current_obs_ind = -1 # index of the latest observation in this batch
    outer_iterations_cnt   = 0     # Total number of iterations performed so far over the outer ifdd,dynmeans loop
    obs_steps               = 0     # NEW for aircraft domain - need to take multiple episodes per obs.
    total_batch_steps       = 0 # Total number of steps observed so far
    total_steps             = 0 # same as total_batch_steps, but not reset for new batch
    force_continued_iterations = False  # If true, then we keep iterating even if both iFDD and dynmeans have converged.

    supportedDomains = ['AircraftField', 'SimpleCar'] # Supported clustering domains.
    # Basically amounts to having a trans_dtype defined [eg single scalar for AircraftField,
    # a 2-D array for actions in SimpleCar, etc.

    def __init__(self,representation,domain,logger,trajLengths,policies,policyNames,
                 trajsPerObs,
                 nRestarts, nExpansions, maxTime, lamb, Q, tau, eta, nu, linear_rep, verbose,
                 JOB_ID,
                 num_obs_between_clustering,
                 maxIterations = -1,
                 numTrueAircraftTypes=-1, # Domain specific, can't get around it.
                 model=None, # For domains requiring observation model
                 force_continued_iterations = False, epsilon = 1e-3,return_best_policy = 0, re_cluster_iterations = 50, use_sparse = False):

        if not className(domain) in self.supportedDomains:
            print 'ERROR: Cannot use ClusterManager Agent with domain ',className(domain)
            print 'Not implemented.  Supported domains are: ',self.supportedDomains

        self.domainName = className(domain)

        self.num_obs_between_clustering = num_obs_between_clustering
        self.epsilon            = epsilon
        self.re_iterations      = re_cluster_iterations
        self.use_sparse         = use_sparse

        self.total_batch_steps  = 0 # Number of steps in this batch of data, ie, sum of observations in this batch
        self.total_steps        = 0
        self.obs_steps          = 0
        self.timestep_cnt       = 0 # Index of timestep of SCORE (incremental) so far

        self.numObs             = len(trajLengths)
        self.numBatches         = ceil(float(self.numObs) / self.num_obs_between_clustering)

        self.trajLengths        = trajLengths
        self.nRestarts          = nRestarts
        self.nExpansions        = nExpansions
        self.maxTime            = maxTime
        self.max_iterations     = maxIterations

        self.aircraft_types     = ones(self.numObs) * -1

        self.policies           = policies
        self.policy_names       = policyNames
        self.policy             = policies[0]

        self.num_true_aircraft_types = numTrueAircraftTypes
        self.start_time     = time()

        if(len(self.policies) != self.numObs):
            logger.log('ERROR: specified different number of policies than trajectories.')
            return None

        self.lamb                   = lamb
        self.eta                    = eta
        # (log since for forcing single-cluster, we use a huge value which can't fit in filename)

        # Make A and r incrementally if the representation can not expand

        numPerformanceChecks = self.max_iterations * ceil(self.numObs / self.num_obs_between_clustering)

        self.outer_iterations_cnt = 0
        self.STATS              = [] # Like policy iteration, don't know how many we need to record.

        self.force_continued_iterations = force_continued_iterations

        super(ClusterManager, self).__init__(representation, self.policies[0], domain,logger)
        if logger:
            self.logger.log('Max clustering iterations for single observation set:\t%d' % self.max_iterations)
            self.logger.log('Number of cluster restarts:\t%d' % nRestarts)
            self.logger.log('Total number of observations:\t%d' % self.numObs)
            self.logger.log('Number of observations between clusterings:\t%d' % self.num_obs_between_clustering)
            self.logger.log('Lambda:\t%.3f' % lamb)
            self.logger.log('Q:\t%.3f' % Q)
            self.logger.log('Tau:\t%.3f' % tau)
            self.logger.log('Eta:\t%.3f' % eta)
            self.logger.log('Number of possible performance checks:\t%d' % numPerformanceChecks)
            self.logger.log('Weight Difference tol.:\t%0.3f' % self.epsilon)
            self.logger.log('Track the best policy:\t%d' % self.return_best_policy)
            self.logger.log('Use Sparse:\t\t%d' % self.use_sparse)

        self.clearOldData() # Must come BEFORE setTransDType
        self.setTransDType()

        featureType = self.representation.featureType()
            # TODO could not figure out how to automatically set feature type below

        self.allocateNextWindow(0) # Allocate our first observation

        self.prev_min_objective = inf
        self.prev_sum_sqerr = inf
        self.prev_representation = None

        self.job_id = JOB_ID

        self.edmHelper          = EDMHelper_score(lamb, Q, tau, eta, nu, linear_rep, verbose, JOB_ID)

    def clearOldData(self):
        self.eps_steps      = 0
        self.total_batch_steps = 0
        self.obs_steps      = 0
        self.prev_sum_sqerr = inf
        prev_min_objective  = inf
        self.transitionStructList = []

    # episode_number corresponds to the 1-indexed episode we are ending
    # [or the zero-indexed episode we are starting]
    # NOTE that episode_number refers to ENTIRE set of observations,
    # not just the batch we are working with - ie, unaffected by clearOldData
    #
    # current_obs_ind refers to the overall data, never reset
    def allocateNextWindow(self, obs_number):
        windowInd1 = int(obs_number)
        windowInd2 = int(obs_number+1)
        self.max_window = 0
        for numTrajTransitions in self.trajLengths[windowInd1:windowInd2]:
            self.max_window += numTrajTransitions
        self.logger.log('Max Data Size for new observation, covers trajs %d to %d:\t\t %d' %
                        (windowInd1,windowInd2,self.max_window))
                    #Take memory for stored values of next observation
                                # Preallocate for storage
        self.transitionStructList.append(empty(self.max_window, dtype=self.trans_dtype))

        self.current_obs_ind += 1
    #########
    # We hit the end of an observation, close it up. episode_number corresponds
    # to the 0-indexed episode we are ending.
    # Returns True if successfully converge
    def endObservation(self, traj_number, premature_ending = False):
        # Remember our hidden state of aircraft_type so we can check true values later.
        self.aircraft_types[self.current_obs_ind] = int(self.domain.getHiddenStates()[0]) # Only one hidden state, aircraft_type

        if premature_ending:
            self.logger.log('WARNING: Premature ending - MUST IMPLEMENT CODE to handle')
            return
            #FIXME traj terminated before desired number of transitions, must resize data matrices

        next_traj_num = traj_number+1
        next_obs_num = self.current_obs_ind + 1
        # Ready to end this observation, collection of trajs
        edmHelper = self.edmHelper
        isFinalObservation = False
        if(next_traj_num >= self.numObs):
            isFinalObservation = True

        if ((next_obs_num) % self.num_obs_between_clustering) == 0 or isFinalObservation:  # Time to re-cluster using accumulated observations
            self.logger.log('Time to cluster on new observations')
            iteration_start_time = time()
#                 self.sortTransitionStructList()

            feature_added       = True
            clusters_changed    = True
            etaThresholdReached = False # If we hit eta, SCORE stops.





#                 legacyInnerClustering()




            firstObsInd = next_obs_num - len(self.transitionStructList)
            obsList = []
            predictions = zeros(self.total_batch_steps)
            curStep = 0 # Step (in flattened sense, across all observations
            trueValue_list = [] # List of arrays of true values at each observation

            actualTypes = array(self.aircraft_types[firstObsInd:firstObsInd+len(self.transitionStructList)], dtype=int)

            for uniqueObsInd, (transStruct, actualType) in enumerate(zip(self.transitionStructList, actualTypes)):
                absoluteObsInd = firstObsInd + uniqueObsInd

                trueValues = array([self.domain.getFunctionVal(s, actualType) for s in self.transitionStructList[uniqueObsInd]['s']])
                trueValue_list.append(trueValues)

                obsList.append(EDMHelper_score.getSwigObs(absoluteObsInd, transStruct, self.domainName))

#                     predictions = EDMHelper_Deviations.vector2Arr(params[label].getPrediction(obs)) # Array of predicted values


#                         cnts = EDMHelper_Deviations.vector2Arr(obs.cnts)
#                         print 'delete me: counts are ', cnts

                    # Keep track of how many aircraft were assigned to each cluster
                    # Rows are true type, columns are label assignments.

            observation_vect = EDMHelper_score.toVector_obs(obsList)

            self.edmHelper.run(observation_vect, self.nRestarts, self.nExpansions, self.max_iterations)

            # Absolute index of observation (ie, without resets between batches)
            (repLog,clusLog,objLog, timeLog, accuracyLog, predictionLog) = self.edmHelper.getLatestRunStats(actualTypes, self.logger)

            print repLog,clusLog,objLog, timeLog, accuracyLog
            for outer_iteration, (featuresNum, numParams, objective, deltaTime, clusterAccuracy, allObsPredictions) in enumerate(zip(repLog,clusLog, objLog, timeLog, accuracyLog, predictionLog)):
#                     print 'labels on outer iteration', outer_iteration,', are ', labels
                trueSqError = 0
                for obsPredictions,trueVals in zip(allObsPredictions, trueValue_list):
                    # On this outer_it, looping over all observations
                    # obsPredictions has type LinearError, needs conversion below
                    trueSqError += EDMHelper_score.getTrueSqError(obsPredictions.toStdVec(), trueVals)
                meanTrueSqErrors = trueSqError / float(self.total_batch_steps)
                self.STATS.append([self.outer_iterations_cnt + outer_iteration,   # number of outer iterations
                  self.total_steps, # Number of transitions observed so far
                  featuresNum,      # Number of features
#                       meanSqError,      ###### Error btwn observed mean and predicted using self param; effectively, how bad is your representation
                  meanTrueSqErrors, ###### Error btwn actual fnctn value (of each observation) and that obtained using their param
                  deltaTime,  # Time since start
                  numParams,        # number of clusters
                  objective,         # Minimum objective value for this iteration
                  clusterAccuracy,
                  self.timestep_cnt
                  ])

            self.outer_iterations_cnt += len(repLog)
            self.timestep_cnt += 1

            # No longer want any of our data
            self.clearOldData()


        if(not isFinalObservation):
            self.allocateNextWindow(next_obs_num)
            self.eps_steps = 0 # We're starting a new episode
            self.obs_steps = 0 # We also finished our observation.
            self.policy = self.policies[next_traj_num]
            self.logger.log('Ended another observation.  Now switching to policy: ** %s **' % self.policy.policyName)
            # The above line will throw error if policyName is not defined,
            # currently only for FixedPolicy
        else:
            self.logger.log('Finished last observation.')

    def learn(self,s,a,r,ns,na,terminal,transLogger=None):
        self.process(s,a,r,ns,na,terminal,transLogger)

    def process(self,s,a,r,ns,na,terminal,transLogger):
        curObs = self.current_obs_ind
        transitionStruct = self.transitionStructList[curObs % self.num_obs_between_clustering]

        cur_step = self.obs_steps

        transitionStruct[cur_step]['sHash']    = self.representation.hashState(s,hiddenIndices=[0])
#         transitionStruct[cur_step]['phiS'][:]     = phi_s
        transitionStruct[cur_step]['s'][:]        = s
#         transitionStruct[cur_step]['ns'][:]       = ns
        transitionStruct[cur_step]['r']        = r

        if self.domainName == 'AircraftField':
            transitionStruct[cur_step]['scalar']    = r

            if transLogger:
                transLogger.log("%d %s %s %f" %(
                transitionStruct[cur_step]['sHash'],
                transitionStruct[cur_step]['s'],
                transitionStruct[cur_step]['scalar'],
                transitionStruct[cur_step]['r']), printToScreen=False)

        elif self.domainName == 'SimpleCar':
            # We pretend we do not have access to the true action taken,
            # and instead compute the action which we THINK got us there.
            expectedAction = model.expectedActionTaken(s0, ns)
            turnInput = expectedAction[1]
            transitionStruct[cur_step]['scalar']    = turnInput

            if transLogger:
                transLogger.log("%d %s %f" %(
                transitionStruct[cur_step]['sHash'],
                transitionStruct[cur_step]['scalar'],
                transitionStruct[cur_step]['r']), printToScreen=False)



        self.eps_steps += 1
        self.obs_steps += 1
        self.total_batch_steps += 1 # Same as eps_steps, but only resets after whole batch is done
        self.total_steps += 1

    def sortTransitionStructList(self, transitionStructList = None):
        pass

    def setTransDType(self):
        if self.domainName == 'AircraftField':
            self.trans_dtype=[('sHash','int32'),
            ('s', '%dint' % (self.domain.state_space_dims)),
            ('phiS', '%dbool'% (self.representation.features_num)),
            ('scalar','float64'),
            ('r','float64')]
        elif self.domainName == 'SimpleCar':
            self.trans_dtype=[('sHash','int32'),
            ('phiS', '%dbool'% (self.representation.features_num)),
            ('scalar','float64'),
            ('r','float64')]
        else:
            print 'ERROR: Attempted to set transition type for unrecognized domain, ',self.domainName

    # Returns a list of the parameter objects associated with clusterer.
    # Almost always called after SCORE has completely finished.
    def getValue(self, sHash, paramInd):
        return self.edmHelper.getValue(sHash, paramInd)
    def getSqDistTo(self, obs, paramInd):
        return self.edmHelper.getSqDistTo(obs, paramInd)

#             ('s','%dfloat64' % self.domain.state_space_dims)]
#                 ('ns','%dfloat64' % self.domain.state_space_dims), ('r','float64')]

    def legacyInnerClustering(self):
        for clustering_it in arange(self.max_iterations):
                    obsList = []
                    for transStruct in self.transitionStructList:
                        obsList.append(EDMHelper_Deviations.getSwigObs_Field(self.current_obs_ind, transStruct, self.representation))
                    observation_vect = EDMHelper_Deviations.toVector_obs(obsList)


#                     paramBufferSize = 100
#                     labelBufferSize = self.numTrajs # Allocate enough for total overfitting
#
#                     finalLabels = edm.vector_param(max_clusters)=
#                     finalParams =

                    # TODO - is it really necessary to preallocate?
                    labels = EDMHelper_Deviations.vector_i(len=5000)
                    params = EDMHelper_Deviations.vector_param(len=5000)

                    ######
                    # Populates finalObj, finalLabels, finalParams
                    minObjective = edmHelper.cluster(observation_vect, self.nRestarts, selectedLabeling, labels, params, self.job_id, self.maxTime)
                    selectedLabeling = labels
                    edmHelper.labels = labels
                    edmHelper.params = params

                    numParams = edmHelper.getNumParams()
                    # Put 'params' in a python data structure
                    params = EDMHelper_Deviations.vector2Arr(params, numParams)

#                     minObjective = edmHelper.cluster(observation_vect, self.nRestarts, self.maxTime) # Cluster over these observations
                    minObjective -= self.lamb # This only makes sense when ages.size() = 0 in extdynmeans

                    if minObjective > self.prev_min_objective: # We will not gain by adding this feature or k-means had error
                        self.logger.log('WARNING: Objective function increased from previous step,')
                        self.logger.log('either from extDynMeans error or because adding feature')
                        self.logger.log('provides no advantage.  Halting.')
#                         break

                    self.prev_min_objective = minObjective
                    if minObjective > 10 ** 300: # There was no min objective
                        print 'WARNING: no minimum objective, setting to arbitrary value.'
                        minObjective = 10 # Arbitrary, recognizable number
                    # FIXME check for timeout - changed dynMeans to give objective fnctn instead
#                     timedOut = False
#                     self.logger.log("Did we time out on cluster iteration %d? %r" % (clustering_it, timedOut))
                    self.outer_iterations_cnt += 1

#                     EDMHelper_Deviations.logTransErrors(observation_vect, params, labels, self.logger)
#                     sumOfSqErrors = edmHelper.getTransSqErrorSum()

                    # Update in case we added new features since last time
#                     self.allTransitionsStruct = self.allTransitionsStruct.astype(self.trans_dtype)
#                     for i,s in enumerate(self.allTransitionsStruct['s']):
#                         self.allTransitionsStruct[i]['phiS'][:] = self.representation.phi_nonTerminal(s)
                    # Assign phiNS based on policy in loop below:


                    sumTrueSqErrors = 0
                    sumOfSqErrors = 0
                    for obs_ind, (obs, label) in enumerate(zip(obsList, labels)):
                        absolute_obs_ind = firstObsInd + obs_ind
                        predictions = EDMHelper_Deviations.vector2Arr(params[label].getPrediction(obs)) # Array of predicted values
#                         cnts = EDMHelper_Deviations.vector2Arr(obs.cnts)
#                         print 'delete me: counts are ', cnts
                        actualType = int(self.aircraft_types[absolute_obs_ind])

                        # Keep track of how many aircraft were assigned to each cluster
                        # Rows are true type, columns are label assignments.
                        if label < self.num_true_aircraft_types:
                            self.clusterBins[actualType, label] += 1

                        trueValues = array([self.domain.getFunctionVal(s, actualType) for s in self.transitionStructList[obs_ind]['s']])
#                         trueValues = trueValues.T
#                         print 'ALL TRUE VALUES OF FUNCTION [DUPLICATES]', trueValues
#                         print 'ALL PREDICTED VALUES OF FUNCTION [DUPLICATES]', predictions

#                         print 'length of trueValues and predictions', len(trueValues), len(predictions)
                        trueErrors = (trueValues - predictions)
                        trueSqErrors = trueErrors ** 2
#                         print 'DIFFERENCE', trueErrors
#                         print 'ALL sqERRORS', trueSqErrors

#                         for (paramID, count) in listOfPolicyParams:
#                         sqTrueParamError += params[paramID].getSqDistTo(allTransitionObs) # for now ignore count, otherwise weight by those
#                             self.logger.log('Policy %s associated with parameter %d' % (policyName, paramID))

#                         sqTrueParamError /= len(listOfPolicyParams) # Normalize by number of params
                        # [ie, the error associated with this policy is the average error
                        # from the parameter you would end up with]
#                         print 'DEBUG 1.trueErrors', trueErrors
#                         print 'DEBUG 2.trueSqErrors', trueSqErrors

                        trueErrors = sum(abs(trueErrors))
                        trueSqErrors = sum(trueSqErrors)

#                         self.logger.log('Error for of observation (performance under its param, whatever it is, vs actual) is ')
#                         self.logger.log('1.Errors btwn actual and computed under param:\t%.3f ' % trueErrors)
#                         self.logger.log('2.Errors btwn actual and computed under param sq:\t%.3f ' % trueSqErrors)
#                         self.logger.log('***Errors btwn actual and computed under param sq weighted:\t%.3f ' % trueSqWeightedErrors)
                        sumTrueSqErrors += trueSqErrors # For now, ignore count, otherwise must divide over sum of those
                        sumOfSqErrors += params[label].getSqDistTo(obs)

                    meanTrueSqErrors = sumTrueSqErrors / self.total_batch_steps
                    meanSqError = sumOfSqErrors / self.total_batch_steps


                    #DEBUG
                    if(self.prev_sum_sqerr < sumOfSqErrors):
                        print 'DEBUG CAUTION: increased sumOfSqErrors'
                        print 'Not necessarily error, but SCORE will halt despite possible future improvement.'
                        print 'Likely an unlucky clustering.'



                    # Note that we compare eta with sum of sq errors, not MSE
                    if(self.prev_sum_sqerr - sumOfSqErrors < self.eta):
                        print 'Failed to reduce sq error sufficiently for eta:'
                        print 'previous sum sq error - (current) sumSqError = '
                        print self.prev_sum_sqerr,' - ',sumOfSqErrors,' = ',self.prev_sum_sqerr - sumOfSqErrors
                        print self.prev_sum_sqerr - sumOfSqErrors,' > ',self.eta
                        etaThresholdReached = True
                        # We don't log any stats, etc., since SCORE refuses this new feature.
                        break


                    self.prev_sum_sqerr = sumOfSqErrors

                    self.STATS.append([self.outer_iterations_cnt,   # number of outer iterations
                                       self.total_steps,       # Number of transitions observed so far
                                  self.representation.features_num, # Number of features
                                  meanSqError,    # Error btwn observed mean and predicted using self param; effectively, how bad is your representation
                                  meanTrueSqErrors,   # Error btwn actual fnctn value (of each observation) and that obtained using their param
                                  deltaT(self.start_time),      # Time since start
                                  numParams,                   # number of clusters
                                  minObjective            # Minimum objective value for this iteration
                                  ])
    #                               0])                           # TEMPORARY return

                    # Store the old labels/params for when we fall out of loop
                    self.prev_labels = labels
                    self.prev_params = params

                    clusters_changed = edmHelper.isClustersChanged()
                    if(clusters_changed):
                        self.logger.log('Re-clustering caused a change in clusters')
        #            errorVec = edmHelper.getObsErrors() # vector of vectors of errors on most recent observations
    #                 print 'num features', self.representation.features_num
    #                 feature_added, STATS  = self.representationExpansionDynMeans()
                    feature_added = False
                    if hasFunction(self.representation,'batchDiscover'):
                        if self.is_incremental:
                            self.prev_representation = self.representation # Keep the old representation around in case SCORE doesn't like your new feature.
                            self.representation = deepcopy(self.representation)
                        feature_added = self.representationExpansionDynMeans()
    #                 print 'new num features',self.representation.features_num

                    if(feature_added): # Must recompute phiS, phiNS
                        self.logger.log('Re-clustering: Added feature(s) on iteration %d' % clustering_it)
    #                     print 'previous num features in struct', (self.transitionStructList[0])['phiS'][0][:]
                        self.updateStructDTypeCnt() # Make room for new feature elements
    #                     print 'new num features in struct', (self.transitionStructList[0])['phiS'][0][:]
    #                 self.logger.log("%d: %0.0f(s), ||w1-w2|| = %0.4f, %+0.3f Return, %d Steps, %d Features" \
    #                                 % (clustering_it,deltaT(iteration_start_time),weight_diff,
    #                                    eps_return,eps_length,self.representation.features_num))

                    if not self.force_continued_iterations:
                        if((not feature_added) and (not clusters_changed)):
                            self.logger.log('Converged Early! Exiting loop at iteration %d of %d' %(clustering_it, self.max_iterations))
                            break
                    # Else we'll recluster even though we know we're done.
