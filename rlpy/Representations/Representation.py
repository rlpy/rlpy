"""Representation base class."""

import logging
from copy import deepcopy
from rlpy.Tools import className, addNewElementForAllActions
from rlpy.Tools import vec2id, bin2state, findElemArray1D
from rlpy.Tools import hasFunction, id2vec, closestDiscretization
import scipy.sparse as sp
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Representation(object):

    """
    The Representation is the :py:class:`~rlpy.Agents.Agent.Agent`'s model of the
    value function associated with a :py:class:`~rlpy.Domains.Domain.Domain`.

    As the Agent interacts with the Domain, it receives updates in the form of
    state, action, reward, next state, next action. \n
    The Agent passes these quantities to its Representation, which is
    responsible for maintaining the value function usually in some
    lower-dimensional feature space.
    Agents can later query the Representation for the value of being in a state
    *V(s)* or the value of taking an action in a particular state
    ( known as the Q-function, *Q(s,a)* ).

    .. note::

        Throughout the framework, ``phi`` refers to the vector of features;
        ``phi`` or ``phi_s`` is thus the vector of feature functions evaluated
        at the state *s*.  phi_s_a appends \|A\|-1 copies of phi_s, such that
        \|phi_s_a\| = \|A\| * \|phi\|, where \|A\| is the size of the action
        space and \|phi\| is the number of features.  Each of these blocks
        corresponds to a state-action pair; all blocks except for the selected
        action ``a`` are set to 0.

    The Representation class is a base class that provides the basic framework
    for all representations. It provides the methods and attributes
    that allow child classes to interact with the Agent and Domain classes
    within the RLPy library. \n
    All new representation implementations should inherit from this class.

    .. note::
        At present, it is assumed that the Linear Function approximator
        family of representations is being used.

    """
    #: A numpy array of the Linear Weights, one for each feature (theta)
    weight_vec = None
    #: The Domain that this Representation is modeling
    domain = None
    #: Number of features in the representation
    features_num = 0
    #: Number of actions in the representation
    actions_num = 0
    # Number of bins used for discretization of each continuous dimension
    discretization = 20
    #: Number of possible states per dimension [1-by-dim]
    bins_per_dim = 0
    #: Width of bins in each dimension
    binWidth_per_dim = 0
    #: Number of aggregated states based on the discretization.
    #: If the represenation is adaptive, set to the best resolution possible
    agg_states_num = 0
    # A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None

    #: True if the number of features may change during execution.
    isDynamic = False
    #: A dictionary used to cache expected results of step(). Used for planning algorithms
    expectedStepCached = None

    def __init__(self, domain, discretization=20, seed=1):
        """
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        """

        for v in ['features_num']:
            if getattr(self, v) is None:
                raise Exception('Missed domain initialization of ' + v)
        self.expectedStepCached = {}
        self.setBinsPerDimension(domain, discretization)
        self.domain = domain
        self.state_space_dims = domain.state_space_dims
        self.actions_num = domain.actions_num
        self.discretization = discretization
        try:
            self.weight_vec = np.zeros(self.features_num * self.actions_num)
        except MemoryError as m:
            print(
                "Unable to allocate weights of size: %d\n" %
                self.features_num *
                self.actions_num)
            raise m

        self._phi_sa_cache = np.empty(
            (self.actions_num, self.features_num))
        self._arange_cache = np.arange(self.features_num)
        self.agg_states_num = np.prod(self.bins_per_dim.astype('uint64'))
        self.logger = logging.getLogger("rlpy.Representations." + self.__class__.__name__)
        
        # a new stream of random numbers for each representation
        self.random_state = np.random.RandomState(seed=seed)
        
    def init_randomization(self):
        """
        Any stochastic behavior in __init__() is broken out into this function
        so that if the random seed is later changed (eg, by the Experiment),
        other member variables and functions are updated accordingly.
        
        """
        pass
        
    def V(self, s, terminal, p_actions, phi_s=None):
        """ Returns the value of state s under possible actions p_actions.

        :param s: The queried state
        :param terminal: Whether or not *s* is a terminal state
        :param p_actions: the set of possible actions
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        See :py:meth:`~rlpy.Representations.Representation.Representation.Qs`.
        """

        if phi_s is None:
            phi_s = self.phi(s, terminal)
        AllQs = self.Qs(s, terminal, phi_s)
        if len(p_actions):
            return max(AllQs[p_actions])
        else:
            return 0  # Return 0 value when no action is possible

    def Qs(self, s, terminal, phi_s=None):
        """
        Returns an array of actions available at a state and their
        associated values.

        :param s: The queried state
        :param terminal: Whether or not *s* is a terminal state
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        :return: The tuple (Q,A) where:
            - Q: an array of Q(s,a), the values of each action at *s*. \n
            - A: the corresponding array of actionIDs (integers)

        .. note::
            This function is distinct
            from :py:meth:`~rlpy.Representations.Representation.Representation.Q`,
            which computes the Q function for an (s,a) pair. \n
            Instead, this function ``Qs()`` computes all Q function values
            (for all possible actions) at a given state *s*.

        """

        if phi_s is None:
            phi_s = self.phi(s, terminal)
        if len(phi_s) == 0:
            return np.zeros((self.actions_num))
        weight_vec_prime = self.weight_vec.reshape(-1, self.features_num)
        if self._phi_sa_cache.shape != (self.actions_num, self.features_num):
            self._phi_sa_cache = np.empty(
                (self.actions_num, self.features_num))
        Q = np.multiply(weight_vec_prime, phi_s,
                        out=self._phi_sa_cache).sum(axis=1)
        # stacks phi_s in cache
        return Q

    def Q(self, s, terminal, a, phi_s=None):
        """ Returns the learned value of a state-action pair, *Q(s,a)*.

        :param s: The queried state in the state-action pair.
        :param terminal: Whether or not *s* is a terminal state
        :param a: The queried action in the state-action pair.
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        :return: (float) the value of the state-action pair (s,a), Q(s,a).

        """
        if len(self.weight_vec) > 0:
            phi_sa, i, j = self.phi_sa(s, terminal, a, phi_s, snippet=True)
            return np.dot(phi_sa, self.weight_vec[i:j])
        else:
            return 0.0

    def phi(self, s, terminal):
        """
        Returns :py:meth:`~rlpy.Representations.Representation.Representation.phi_nonTerminal`
        for a given representation, or a zero feature vector in a terminal state.

        :param s: The state for which to compute the feature vector

        :return: numpy array, the feature vector evaluted at state *s*.

        .. note::
            If state *s* is terminal the feature vector is returned as zeros!
            This prevents the learning algorithm from wrongfully associating
            the end of one episode with the start of the next (e.g., thinking
            that reaching the terminal state causes it to teleport back to the
            start state s0).


        """
        if terminal or self.features_num == 0:
            return np.zeros(self.features_num, 'bool')
        else:
            return self.phi_nonTerminal(s)

    def phi_sa(self, s, terminal, a, phi_s=None, snippet=False):
        """
        Returns the feature vector corresponding to a state-action pair.
        We use the copy paste technique (Lagoudakis & Parr 2003).
        Essentially, we append the phi(s) vector to itself *|A|* times, where
        *|A|* is the size of the action space.
        We zero the feature values of all of these blocks except the one
        corresponding to the actionID *a*.

        When ``snippet == False`` we construct and return the full, sparse phi_sa.
        When ``snippet == True``, we return the tuple (phi_s, index1, index2)
        where index1 and index2 are the indices defining the ends of the phi_s
        block which WOULD be nonzero if we were to construct the full phi_sa.

        :param s: The queried state in the state-action pair.
        :param terminal: Whether or not *s* is a terminal state
        :param a: The queried action in the state-action pair.
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.
        :param snippet: if ``True``, do not return a single phi_sa vector,
            but instead a tuple of the components needed to create it.
            See return value below.

        :return: If ``snippet==False``, return the enormous phi_sa vector
            constructed by the copy-paste method.
            If ``snippet==True``, do not construct phi_sa, only return
            a tuple (phi_s, index1, index2) as described above.

        """
        if phi_s is None:
            phi_s = self.phi(s, terminal)
        if snippet is True:
            return phi_s, a * self.features_num, (a + 1) * self.features_num

        phi_sa = np.zeros(
            (self.features_num * self.actions_num),
            dtype=phi_s.dtype)
        if self.features_num == 0:
            return phi_sa
        if len(self._arange_cache) != self.features_num:
            self._arange_cache = np.arange(
                a * self.features_num,
                (a + 1) * self.features_num)
        else:
            self._arange_cache += a * self.features_num - self._arange_cache[0]
        phi_sa[self._arange_cache] = phi_s
        # Slower alternatives
        # Alternative 1: Set only non_zeros (Very close on running time with the current solution. In fact it is sometimes better)
        #nnz_ind = phi_s.nonzero()
        #phi_sa[nnz_ind+a*self.features_num] = phi_s[nnz_ind]
        # Alternative 2: Use of Kron
        #A = zeros(self.actions_num)
        #A[a] = 1
        #F_sa = kron(A,F_s)
        return phi_sa

    def addNewWeight(self):
        """
        Add a new zero weight, corresponding to a newly added feature,
        to all actions.
        """
        self.weight_vec = addNewElementForAllActions(
            self.weight_vec,
            self.actions_num)

    def hashState(self, s,):
        """
        Returns a unique id for a given state.
        Essentially, enumerate all possible states and return the ID associated
        with *s*.

        Under the hood: first, discretize continuous dimensions into bins
        as necessary. Then map the binstate to an integer.
        """
        ds = self.binState(s)
        return vec2id(ds, self.bins_per_dim)

    def setBinsPerDimension(self, domain, discretization):
        """
        Set the number of bins for each dimension of the domain.
        Continuous spaces will be slices using the ``discretization`` parameter.
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param discretization: The number of bins a continuous domain should be sliced into.

        """
        self.bins_per_dim = np.zeros(domain.state_space_dims, np.uint16)
        self.binWidth_per_dim = np.zeros(domain.state_space_dims)
        for d in xrange(domain.state_space_dims):
            if d in domain.continuous_dims:
                self.bins_per_dim[d] = discretization
            else:
                self.bins_per_dim[d] = domain.statespace_limits[d, 1] - \
                    domain.statespace_limits[d, 0]
            self.binWidth_per_dim[d] = (domain.statespace_limits[d,1] - domain.statespace_limits[d, 0]) / (self.bins_per_dim[d] * 1.)

    def binState(self, s):
        """
        Returns a vector where each element is the zero-indexed bin number
        corresponding with the given state.
        (See :py:meth:`~rlpy.Representations.Representation.Representation.hashState`)
        Note that this vector will have the same dimensionality as *s*.

        (Note: This method is binary compact; the negative case of binary features is
        excluded from feature activation.
        For example, if the domain has a light and the light is off, no feature
        will be added. This is because the very *absence* of the feature
        itself corresponds to the light being off.
        """
        s = np.atleast_1d(s)
        limits = self.domain.statespace_limits
        assert (np.all(s >= limits[:, 0]))
        assert (np.all(s <= limits[:, 1]))
        width = limits[:, 1] - limits[:, 0]
        diff = s - limits[:, 0]
        bs = (diff * self.bins_per_dim / width).astype("uint32")
        m = bs == self.bins_per_dim
        bs[m] = self.bins_per_dim[m] - 1
        return bs

    def bestActions(self, s, terminal, p_actions, phi_s=None):
        """
        Returns a list of the best actions at a given state.
        If *phi_s* [the feature vector at state *s*] is given, it is used to
        speed up code by preventing re-computation within this function.

        See :py:meth:`~rlpy.Representations.Representation.Representation.bestAction`

        :param s: The given state
        :param terminal: Whether or not the state *s* is a terminal one.
        :param phi_s: (optional) the feature vector at state (s).
        :return: A list of the best actions at the given state.

        """
        Qs = self.Qs(s, terminal, phi_s)
        Qs = Qs[p_actions]
        # Find the index of best actions
        ind = findElemArray1D(Qs, Qs.max())
        return np.array(p_actions)[ind]

    def pre_discover(self, s, terminal, a, sn, terminaln):
        """
        Identifies and adds ("discovers") new features for this adaptive
        representation BEFORE having obtained the TD-Error.
        For example, see :py:class:`~rlpy.Representations.IncrementalTabular.IncrementalTabular`.
        In that class, a new feature is added anytime a novel state is observed.

        .. note::
            For adaptive representations that require access to TD-Error to
            determine which features to add next,
            use :py:meth:`~rlpy.Representations.Representation.Representation.post_discover`
            instead.

        :param s: The state
        :param terminal: boolean, whether or not *s* is a terminal state.
        :param a: The action
        :param sn: The next state
        :param terminaln: boolean, whether or not *sn* is a terminal state.

        :return: The number of new features added to the representation
        """

        return 0

    def post_discover(self, s, terminal, a, td_error, phi_s):
        """
        Identifies and adds ("discovers") new features for this adaptive
        representation AFTER having obtained the TD-Error.
        For example, see :py:class:`~rlpy.Representations.iFDD.iFDD`.
        In that class, a new feature is added based on regions of high TD-Error.

        .. note::
            For adaptive representations that do not require access to TD-Error
            to determine which features to add next, you may
            use :py:meth:`~rlpy.Representations.Representation.Representation.pre_discover`
            instead.

        :param s: The state
        :param terminal: boolean, whether or not *s* is a terminal state.
        :param a: The action
        :param td_error: The temporal difference error at this transition.
        :param phi_s: The feature vector evaluated at state *s*.

        :return: The number of new features added to the representation
        """
        return 0

    def bestAction(self, s, terminal, p_actions, phi_s=None):
        """
        Returns the best action at a given state.
        If there are multiple best actions, this method selects one of them
        uniformly randomly.
        If *phi_s* [the feature vector at state *s*] is given, it is used to
        speed up code by preventing re-computation within this function.

        See :py:meth:`~rlpy.Representations.Representation.Representation.bestActions`

        :param s: The given state
        :param terminal: Whether or not the state *s* is a terminal one.
        :param phi_s: (optional) the feature vector at state (s).
        :return: The best action at the given state.
        """
        bestA = self.bestActions(s, terminal, p_actions, phi_s)
        if isinstance(bestA, int):
            return bestA
        elif len(bestA) > 1:
            return self.random_state.choice(bestA)
            # return bestA[0]
        else:
            return bestA[0]

    def phi_nonTerminal(self, s):
        """ *Abstract Method* \n
        Returns the feature vector evaluated at state *s* for non-terminal
        states; see
        function :py:meth:`~rlpy.Representations.Representation.Representation.phi`
        for the general case.

        :param s: The given state

        :return: The feature vector evaluated at state *s*.
        """
        raise NotImplementedError

    def activeInitialFeatures(self, s):
        """
        Returns the index of active initial features based on bins in each
        dimension.
        :param s: The state

        :return: The active initial features of this representation
            (before expansion)
        """
        bs = self.binState(s)
        shifts = np.hstack((0, np.cumsum(self.bins_per_dim)[:-1]))
        index = bs + shifts
        return index.astype('uint32')

    def batchPhi_s_a(self, all_phi_s, all_actions,
                     all_phi_s_a=None, use_sparse=False):
        """
        Builds the feature vector for a series of state-action pairs (s,a)
        using the copy-paste method.

        .. note::
            See :py:meth:`~rlpy.Representations.Representation.Representation.phi_sa`
            for more information.

        :param all_phi_s: The feature vectors evaluated at a series of states.
            Has dimension *p* x *n*, where *p* is the number of states
            (indexed by row), and *n* is the number of features.
        :param all_actions: The set of actions corresponding to each feature.
            Dimension *p* x *1*, where *p* is the number of states included
            in this batch.
        :param all_phi_s_a: (Optional) Feature vector for a series of
            state-action pairs (s,a) using the copy-paste method.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.
        :param use_sparse: Determines whether or not to use sparse matrix
            libraries provided with numpy.


        :return: all_phi_s_a (of dimension p x (s_a) )
        """
        p, n = all_phi_s.shape
        a_num = self.actions_num
        if use_sparse:
            phi_s_a = sp.lil_matrix(
                (p, n * a_num), dtype=all_phi_s.dtype)
        else:
            phi_s_a = np.zeros((p, n * a_num), dtype=all_phi_s.dtype)

        for i in xrange(a_num):
            rows = np.where(all_actions == i)[0]
            if len(rows):
                phi_s_a[rows, i * n:(i + 1) * n] = all_phi_s[rows,:]
        return phi_s_a

    def batchBestAction(self, all_s, all_phi_s,
                        action_mask=None, useSparse=True):
        """
        Accepts a batch of states, returns the best action associated with each.

        .. note::
            See :py:meth:`~rlpy.Representations.Representation.Representation.bestAction`

        :param all_s: An array of all the states to consider.
        :param all_phi_s: The feature vectors evaluated at a series of states.
            Has dimension *p* x *n*, where *p* is the number of states
            (indexed by row), and *n* is the number of features.
        :param action_mask: (optional) a *p* x *|A|* mask on the possible
            actions to consider, where *|A|* is the size of the action space.
            The mask is a binary 2-d array, where 1 indicates an active mask
            (action is unavailable) while 0 indicates a possible action.
        :param useSparse: Determines whether or not to use sparse matrix
            libraries provided with numpy.

        :return: An array of the best action associated with each state.

        """
        p, n = all_phi_s.shape
        a_num = self.actions_num

        if action_mask is None:
            action_mask = np.ones((p, a_num))
            for i, s in enumerate(all_s):
                action_mask[i, self.domain.possibleActions(s)] = 0

        a_num = self.actions_num
        if useSparse:
            # all_phi_s_a will be ap-by-an
            all_phi_s_a = sp.kron(np.eye(a_num, a_num), all_phi_s)
            all_q_s_a = all_phi_s_a * self.weight_vec.reshape(-1, 1)  # ap-by-1
        else:
            # all_phi_s_a will be ap-by-an
            all_phi_s_a = np.kron(np.eye(a_num, a_num), all_phi_s)
            all_q_s_a = np.dot(all_phi_s_a, self.weight_vec.T)  # ap-by-1
        all_q_s_a = all_q_s_a.reshape((a_num, -1)).T  # a-by-p
        all_q_s_a = np.ma.masked_array(all_q_s_a, mask=action_mask)
        best_action = np.argmax(all_q_s_a, axis=1)

        # Calculate the corresponding phi_s_a
        phi_s_a = self.batchPhi_s_a(
            all_phi_s,
            best_action,
            all_phi_s_a,
            useSparse)
        return best_action, phi_s_a, action_mask

    def featureType(self):
        """ *Abstract Method* \n
        Return the data type for the underlying features (eg 'float').
        """
        raise NotImplementedError

    def Q_oneStepLookAhead(self, s, a, ns_samples, policy=None):
        """
        Returns the state action value, Q(s,a), by performing one step
        look-ahead on the domain.

        .. note::
            For an example of how this function works, see
            `Line 8 of Figure 4.3 <http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html>`_
            in Sutton and Barto 1998.

        If the domain does not define ``expectedStep()``, this function uses
        ``ns_samples`` samples to estimate the one_step look-ahead.
        If a policy is passed (used in the policy evaluation), it is used to
        generate the action for the next state.
        Otherwise the best action is selected.

        .. note::
            This function should not be called in any RL algorithms unless
            the underlying domain is an approximation of the true model.

        :param s: The given state
        :param a: The given action
        :param ns_samples: The number of samples used to estimate the one_step look-ahead.
        :param policy: (optional) Used to select the action in the next state
            (*after* taking action a) when estimating the one_step look-aghead.
            If ``policy == None``, the best action will be selected.

        :return: The one-step lookahead state-action value, Q(s,a).
        """
        # Hash new state for the incremental tabular case
        self.continuous_state_starting_samples = 10
        if hasFunction(self, 'addState'):
            self.addState(s)

        discount_factor = self.domain.discount_factor
        if hasFunction(self.domain, 'expectedStep'):
            p, r, ns, t, p_actions = self.domain.expectedStep(s, a)
            Q = 0
            for j in xrange(len(p)):
                if policy is None:
                    Q += p[j, 0] * (r[j, 0] + discount_factor * self.V(ns[j,:], t[j,:], p_actions[j]))
                else:
                    # For some domains such as blocks world, you may want to apply bellman backup to impossible states which may not have any possible actions.
                    # This if statement makes sure that there exist at least
                    # one action in the next state so the bellman backup with
                    # the fixed policy is valid
                    if len(self.domain.possibleActions(ns[j,:])):
                        na = policy.pi(ns[j,:], t[j,:], self.domain.possibleActions(ns[j,:]))
                        Q += p[j, 0] * (r[j, 0] + discount_factor * self.Q(ns[j,:], t[j,:], na))
        else:
            # See if they are in cache:
            key = tuple(np.hstack((s, [a])))
            cacheHit = self.expectedStepCached.get(key)
            if cacheHit is None:
                # Not found in cache => Calculate and store in cache
                # If continuous domain, sample <continuous_state_starting_samples> points within each discritized grid and sample <ns_samples>/<continuous_state_starting_samples> for each starting state.
                # Otherwise take <ns_samples> for the state.

                # First put s in the middle of the grid:
                # shout(self,s)
                s = self.stateInTheMiddleOfGrid(s)
                # print "After:", shout(self,s)
                if len(self.domain.continuous_dims):
                    next_states = np.empty(
                        (ns_samples, self.domain.state_space_dims))
                    rewards = np.empty(ns_samples)
                    # next states per samples initial state
                    ns_samples_ = ns_samples / \
                        self.continuous_state_starting_samples
                    for i in xrange(self.continuous_state_starting_samples):
                        # sample a random state within the grid corresponding
                        # to input s
                        new_s = s.copy()
                        for d in xrange(self.domain.state_space_dims):
                            w = self.binWidth_per_dim[d]
                            # Sample each dimension of the new_s within the
                            # cell
                            new_s[d] = (self.random_state.rand() - .5) * w + s[d]
                            # If the dimension is discrete make make the
                            # sampled value to be int
                            if not d in self.domain.continuous_dims:
                                new_s[d] = int(new_s[d])
                        # print new_s
                        ns, r = self.domain.sampleStep(new_s, a, ns_samples_)
                        next_states[i * ns_samples_:(i + 1) * ns_samples_,:] = ns
                        rewards[i * ns_samples_:(i + 1) * ns_samples_] = r
                else:
                    next_states, rewards = self.domain.sampleStep(
                        s, a, ns_samples)
                self.expectedStepCached[key] = [next_states, rewards]
            else:
                # print "USED CACHED"
                next_states, rewards = cacheHit
            if policy is None:
                Q = np.mean([rewards[i] + discount_factor * self.V(next_states[i,:]) for i in xrange(ns_samples)])
            else:
                Q = np.mean([rewards[i] + discount_factor * self.Q(next_states[i,:], policy.pi(next_states[i,:])) for i in xrange(ns_samples)])
        return Q

    def Qs_oneStepLookAhead(self, s, ns_samples, policy=None):
        """
        Returns an array of actions and their associated values Q(s,a),
        by performing one step look-ahead on the domain for each of them.

        .. note::
            For an example of how this function works, see
            `Line 8 of Figure 4.3 <http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html>`_
            in Sutton and Barto 1998.

        If the domain does not define ``expectedStep()``, this function uses
        ``ns_samples`` samples to estimate the one_step look-ahead.
        If a policy is passed (used in the policy evaluation), it is used to
        generate the action for the next state.
        Otherwise the best action is selected.

        .. note::
            This function should not be called in any RL algorithms unless
            the underlying domain is an approximation of the true model.

        :param s: The given state
        :param ns_samples: The number of samples used to estimate the one_step look-ahead.
        :param policy: (optional) Used to select the action in the next state
            (*after* taking action a) when estimating the one_step look-aghead.
            If ``policy == None``, the best action will be selected.

        :return: an array of length `|A|` containing the *Q(s,a)* for each
            possible *a*, where `|A|` is the number of possible actions from state *s*
        """
        actions = self.domain.possibleActions(s)
        Qs = np.array([self.Q_oneStepLookAhead(s, a, ns_samples, policy)
                       for a in actions])
        return Qs, actions

    def V_oneStepLookAhead(self, s, ns_samples):
        """
        Returns the value of being in state *s*, V(s),
        by performing one step look-ahead on the domain.

        .. note::
            For an example of how this function works, see
            `Line 6 of Figure 4.5 <http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html>`_
            in Sutton and Barto 1998.

        If the domain does not define ``expectedStep()``, this function uses
        ``ns_samples`` samples to estimate the one_step look-ahead.

        .. note::
            This function should not be called in any RL algorithms unless
            the underlying domain is an approximation of the true model.

        :param s: The given state
        :param ns_samples: The number of samples used to estimate the one_step look-ahead.

        :return: The value of being in state *s*, *V(s)*.
        """
    # The estimated value = max_a Q(s,a) together with the corresponding
    # action that maximizes the Q function
        Qs, actions = self.Qs_oneStepLookAhead(s, ns_samples)
        a_ind = np.argmax(Qs)
        return Qs[a_ind], actions[a_ind]

    def stateID2state(self, s_id):
        """
        Returns the state vector correponding to a state_id.
        If dimensions are continuous it returns the state representing the
        middle of the bin (each dimension is discretized according to
        ``representation.discretization``.

        :param s_id: The id of the state, often calculated using the
            ``state2bin`` function

        :return: The state *s* corresponding to the integer *s_id*.
        """

        # Find the bin number on each dimension
        s = np.array(id2vec(s_id, self.bins_per_dim))

        # Find the value corresponding to each bin number
        for d in xrange(self.domain.state_space_dims):
            s[d] = bin2state(s[d], self.bins_per_dim[d], self.domain.statespace_limits[d,:])

        if len(self.domain.continuous_dims) == 0:
            s = s.astype(int)
        return s

    def stateInTheMiddleOfGrid(self, s):
        """
        Accepts a continuous state *s*, bins it into the discretized domain,
        and returns the state of the nearest gridpoint.
        Essentially, we snap *s* to the nearest gridpoint and return that
        gridpoint state.
        For continuous MDPs this plays a major rule in improving the speed
        through caching of next samples.

        :param s: The given state

        :return: The nearest state *s* which is captured by the discretization.
        """
        s_normalized = s.copy()
        for d in xrange(self.domain.state_space_dims):
            s_normalized[d] = closestDiscretization(s[d], self.bins_per_dim[d], self.domain.statespace_limits[d,:])
        return s_normalized


    def episodeTerminated(self):
        pass
    
    def featureLearningRate(self):
        """
        :return: An array or scalar used to adapt the learning rate of each
        feature individually.
        """
        return 1.0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k is "logger":
                continue
            setattr(result, k, deepcopy(v, memo))
        return result
