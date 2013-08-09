import Domains
import numpy as np
import inspect
from nose.tools import ok_, eq_
def test_random_trajectory():
    for d in Domains.__dict__.values():
        if d == Domains.Domain:
            continue
        if inspect.isclass(d) and issubclass(d, Domains.Domain):
            yield check_random_trajectory, d

def test_specification():
    for d in Domains.__dict__.values():
        if d == Domains.Domain:
            continue
        if inspect.isclass(d) and issubclass(d, Domains.Domain):
            yield check_specifications, d

def check_random_trajectory(domain_class):
    """
    run the domain 1000 steps with random actions
    Just make sure no error occurs
    """
    domain = domain_class()
    terminal    = True
    steps       = 0
    T = 1000
    while steps < T:
        if terminal:
            s = domain.s0()
        elif steps % domain.episodeCap == 0:
            s = domain.s0()
        a = np.random.choice(domain.possibleActions(s))
        r,s,terminal = domain.step(a)
        steps += 1

def check_specifications(domain_class):
    domain = domain_class()
    for v in ['statespace_limits', 'actions_num', 'episodeCap']:
        ok_(getattr(domain, v) != None)
