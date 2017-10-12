from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
import pickle
from nose.tools import eq_
from rlpy.Tools import __rlpy_location__
import os


def test_cartpole():
    try:
        from rlpy.Domains import InfCartPoleBalance
    except ImportError:
        print("use old Cartpole class!")
        from rlpy.Domains import Pendulum_InvertedBalance as InfCartPoleBalance

    yield check_traj, InfCartPoleBalance, os.path.join(
        __rlpy_location__,"..", "tests",
        "traj_InfiniteCartpoleBalance.pck")
    try:
        from rlpy.Domains import FiniteCartPoleBalanceOriginal
    except ImportError:
        print("use old Cartpole class!")
        from rlpy.Domains import CartPoleBalanceOriginal as FiniteCartPoleBalanceOriginal

    yield check_traj, FiniteCartPoleBalanceOriginal, os.path.join(
        __rlpy_location__, "..","tests",
        "traj_FiniteCartpoleBalanceOriginal.pck")


def check_traj(domain_class, filename):
    with open(filename) as f:
        traj = pickle.load(f)
    traj_now = sample_random_trajectory(domain_class)
    for i, e1, e2 in zip(list(range(len(traj_now))), traj_now, traj):
        print(i)
        print(e1[0], e2[0])
        if not np.allclose(e1[0], e2[0]):  # states
            print(e1[0], e2[0])
            assert False
        eq_(e1[-1], e2[-1])  # reward
        print("Terminal", e1[1], e2[1])
        eq_(e1[1], e2[1])  # terminal
        eq_(len(e1[2]), len(e2[2]))
        assert np.all([a == b for a, b in zip(e1[2], e2[2])])  # p_actions


def save_trajectory(domain_class, filename):
    traj = sample_random_trajectory(domain_class)
    with open(filename, "w") as f:
        pickle.dump(traj, f)


def sample_random_trajectory(domain_class):
    """
    sample a trajectory of 1000 steps
    """
    traj = []
    np.random.seed(1)
    domain = domain_class()
    domain.random_state = np.random.RandomState(1)
    terminal = True
    steps = 0
    T = 1000
    r = 0
    while steps < T:
        if terminal:
            s, terminal, p_actions = domain.s0()
        elif steps % domain.episodeCap == 0:
            s, terminal, p_actions = domain.s0()
        a = np.random.choice(p_actions)
        traj.append((s, terminal, p_actions, a, r))
        r, s, terminal, p_actions = domain.step(a)
        steps += 1
    return traj
