import rlpy
import numpy as np
from hyperopt import hp

param_space = {
    'kernel_resolution':
    hp.loguniform("kernel_resolution", np.log(5), np.log(50)),
    'discover_threshold':
    hp.loguniform(
        "discover_threshold",
        np.log(1e4),
        np.log(1e8)),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=88044.,
        boyan_N0=64502,
        lambda_=0.43982644088,
        initial_learn_rate=0.920244401,
        kernel_resolution=11.6543336229):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 30
    opt["checks_per_policy"] = 1
    active_threshold = 0.01
    max_base_feat_sim = 0.5
    sparsify = 1

    domain = rlpy.Domains.BicycleRiding()
    opt["domain"] = domain
    kernel_width = (domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]) \
        / kernel_resolution
    representation = rlpy.Representations.KernelizediFDD(domain, sparsify=sparsify,
                               kernel=rlpy.Representations.linf_triangle_kernel,
                               kernel_args=[kernel_width],
                               active_threshold=active_threshold,
                               discover_threshold=discover_threshold,
                               normalization=True,
                               max_active_base_feat=10,
                               max_base_feat_sim=max_base_feat_sim)
    policy = rlpy.Policies.eGreedy(representation, epsilon=0.1)
    # agent           = SARSA(representation,policy,domain,initial_learn_rate=initial_learn_rate,
    # lambda_=.0, learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    opt["agent"] = rlpy.Agents.Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                       lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                       learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = rlpy.Experiments.Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True,
                   visualize_performance=True)
    experiment.plot()
    # experiment.save()
