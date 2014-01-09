from Tools import Logger
from Domains import PuddleWorld 
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Policies import eGreedy
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment
from Policies import BasicPuddlePolicy
import numpy as np
from hyperopt import hp

param_space = {'discover_threshold': hp.loguniform("discover_threshold",
                   np.log(1e-3), np.log(1e2)),
               'discretization': hp.qloguniform('discretization', np.log(5),
                   np.log(50), 1),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/ifdd/",
                    discover_threshold=2.0953701,
                    lambda_=0.58488,
                    boyan_N0=40172.71,
                    discretization=28,
                    initial_alpha=0.17978):
    logger = Logger()
    max_steps = 1000000
    sparsify = 1
    ifddeps = 1e-6
    domain = PuddleWorld(logger=logger)
    pol = BasicPuddlePolicy(domain, None)
    initial_rep = IndependentDiscretization(domain, logger,
                                            discretization=discretization)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          useCache=True,
                          iFDDPlus=1-ifddeps)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    #experiment.plot()
    #experiment.save()
