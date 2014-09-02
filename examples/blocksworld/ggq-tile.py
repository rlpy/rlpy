from rlpy.Domains import BlocksWorld
from rlpy.Agents import Greedy_GQ
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=14.44946,
        initial_learn_rate=0.240155681):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 1
    domain = BlocksWorld(blocks=6, noise=0.3, )
    opt["domain"] = domain
    mat = np.matrix("""1 1 1 0 0 0;
                    0 1 1 1 0 0;
                    0 0 1 1 1 0;
                    0 0 0 1 1 1;

                    0 0 1 0 1 1;
                    0 0 1 1 0 1;
                    1 0 1 1 0 0;
                    1 0 1 0 1 0;
                    1 0 0 1 1 0;
                    1 0 0 0 1 1;
                    1 0 1 0 0 1;
                    1 0 0 1 0 1;
                    1 1 0 1 0 0;
                    1 1 0 0 1 0;
                    1 1 0 0 0 1;
                    0 1 0 1 1 0;
                    0 1 0 0 1 1;
                    0 1 0 1 0 1;
                    0 1 1 0 1 0;
                    0 1 1 0 0 1""")
    #assert(mat.shape[0] == 20)
    representation = TileCoding(
        domain, memory=2000, num_tilings=[1] * mat.shape[0],
        resolution_matrix=mat * 6, safety="none")
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Greedy_GQ(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    # experiment.plot()
    # experiment.save()
