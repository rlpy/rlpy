from rlpy.Representations.Representation import Representation
import numpy as np
import logging
from rlpy.Policies import eGreedy
from rlpy.Agents import SARSA, Q_Learning


class MockRepresentation(Representation):
    def __init__(self):
        """
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        """

        for v in ['features_num']:
            if getattr(self, v) is None:
                raise Exception('Missed domain initialization of ' + v)
        self.expectedStepCached = {}
        self.state_space_dims = 1
        self.actions_num = 1
        self.discretization = 3
        self.features_num = 4
        try:
            self.weight_vec = np.zeros(self.features_num * self.actions_num)
        except MemoryError as m:
            print(
                "Unable to allocate weights of size: %d\n" %
                self.features_num *
                self.domain.actions_num)
            raise m

        self._phi_sa_cache = np.empty(
            (self.actions_num, self.features_num))
        self._arange_cache = np.arange(self.features_num)
        self.logger = logging.getLogger(self.__class__.__name__)

    def phi_nonTerminal(self, s):
        ret = np.zeros(self.features_num)
        ret[s[0]] = 1.
        return ret


def test_sarsa_valfun_chain():
    """
        Check if SARSA computes the value function of a simple Markov chain correctly.
        This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = SARSA(pol, rep, 0.9, lambda_=0.)
    for i in xrange(1000):
        if i % 4 == 3:
            continue
        agent.learn(np.array([i % 4]), [0], 0, 1., np.array([(i + 1) % 4]), [0, ], 0, (i + 2) % 4 == 0)
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_sarsalambda_valfun_chain():
    """
        Check if SARSA(lambda) computes the value function of a simple Markov chain correctly.
        This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = SARSA(pol, rep, 0.9, lambda_=0.5)
    for i in xrange(1000):
        if i % 4 == 3:
            agent.episodeTerminated()
            continue
        agent.learn(np.array([i % 4]), [0], 0, 1., np.array([(i + 1) % 4]), [0], 0, (i + 2) % 4 == 0)
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_qlearn_valfun_chain():
    """
        Check if SARSA computes the value function of a simple Markov chain correctly.
        This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = Q_Learning(pol, rep, 0.9, lambda_=0.)
    for i in xrange(1000):
        if i % 4 == 3:
            continue
        agent.learn(np.array([i % 4]), [0], 0, 1., np.array([(i + 1) % 4]), [0], 0, (i + 2) % 4 == 0)
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_qlambda_valfun_chain():
    """
        Check if SARSA(lambda) computes the value function of a simple Markov chain correctly.
        This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = Q_Learning(pol, rep, 0.9, lambda_=0.5)
    for i in xrange(1000):
        if i % 4 == 3:
            agent.episodeTerminated()
            continue
        agent.learn(np.array([i % 4]), [0], 0, 1., np.array([(i + 1) % 4]), [0], 0, (i + 2) % 4 == 0)
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)
