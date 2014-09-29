"""Functions to be used with hyperopt for doing hyper parameter optimization."""

import os
import rlpy.Tools.results as tres
import rlpy.Tools.run as rt
import hyperopt
import numpy as np
import time
import pickle

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


def dummy_f():
    pass


def _search_condor_parallel(path, space, trials_per_point, setting,
                            objective, max_evals,
                            algo=hyperopt.tpe.suggest,
                            max_queue_len=10, poll_interval_secs=30):
    """
    block_until_done  means that the process blocks until ALL jobs in
    trials are not in running or new state

    suggest() can pass instance of StopExperiment to break out of
    enqueuing loop
    """

    trials = CondorTrials(path=path, ids=range(1, trials_per_point + 1),
                          setting=setting, objective=objective)
    domain = hyperopt.Domain(dummy_f, space, rseed=123)
    trial_path = os.path.join(path, "trials.pck")
    if os.path.exists(trial_path):
        with open(trial_path) as f:
            old_trials = pickle.load(f)
        print "Loaded existing trials"
        if old_trials.setting == trials.setting and trials.ids == old_trials.ids:
            trials = old_trials
    n_queued = trials.count_by_state_unsynced(hyperopt.JOB_STATES)

    def get_queue_len():
        trials.count_by_state_unsynced(hyperopt.base.JOB_STATE_NEW)
        return trials.update_trials(trials._trials)
    stopped = False
    while n_queued < max_evals:
        qlen = get_queue_len()
        while qlen < max_queue_len and n_queued < max_evals:
            n_to_enqueue = 1  # min(self.max_queue_len - qlen, N - n_queued)
            new_ids = trials.new_trial_ids(n_to_enqueue)
            trials.refresh()
            new_trials = algo(new_ids, domain, trials)
            if new_trials is hyperopt.base.StopExperiment:
                stopped = True
                break
            else:
                assert len(new_ids) >= len(new_trials)
                if len(new_trials):
                    trials.insert_trial_docs(new_trials)
                    trials.refresh()
                    n_queued += len(new_trials)
                    qlen = get_queue_len()
                else:
                    break

        with open(trial_path, 'w') as f:
            pickle.dump(trials, f)
        # -- wait for workers to fill in the trials
        time.sleep(poll_interval_secs)
        if stopped:
            break
    while trials.count_by_state_unsynced(hyperopt.base.JOB_STATE_NEW) > 0:
        time.sleep(poll_interval_secs)
    trials.refresh()
    return trials


class CondorTrials(hyperopt.Trials):

    """
    modified trail class specifically designed to run RLPy experiments
    in parallel on a htcondor job scheduling system
    """
    async = True

    def __init__(self, setting, path, ids, objective, **kwargs):
        super(CondorTrials, self).__init__(**kwargs)
        self.path = path
        self.ids = ids
        self.setting = setting
        self.objective = objective

    def refresh(self):
        self.update_trials(self._dynamic_trials)
        super(CondorTrials, self).refresh()

    def _insert_trial_docs(self, docs):
        """insert with no error checking
        """
        rval = [doc['tid'] for doc in docs]

        # submit all jobs to the cluster
        self.update_trials(docs)

        self._dynamic_trials.extend(docs)
        return rval

    def count_by_state_synced(self, arg, trials=None):
        """
        Return trial counts by looking at self._trials
        """
        if trials is None:
            trials = self._trials
        self.update_trials(trials)
        if arg in hyperopt.JOB_STATES:
            queue = [doc for doc in trials if doc['state'] == arg]
        elif hasattr(arg, '__iter__'):
            states = set(arg)
            assert all([x in hyperopt.JOB_STATES for x in states])
            queue = [doc for doc in trials if doc['state'] in states]
        else:
            raise TypeError(arg)
        rval = len(queue)
        return rval

    def unwrap_hyperparam(self, vals):
        return {a: b[0] for a, b in vals.items()}

    def make_full_path(self, hyperparam):
        return (
            os.path.join(
                self.path,
                "-".join([str(v) for v in hyperparam.values()]))
        )

    def update_trials(self, trials):
        count = 0
        for trial in trials:
            if trial["state"] == hyperopt.JOB_STATE_NEW:
                if "submitted" not in trial or not trial["submitted"]:
                    # submit jobs and set status to running
                    hyperparam = self.unwrap_hyperparam(trial["misc"]["vals"])
                    full_path = self.make_full_path(hyperparam)
                    rt.run(self.setting, location=full_path, ids=self.ids,
                           parallelization="condor", force_rerun=False, block=False,
                           **hyperparam)
                    trial["submitted"] = True
                else:
                    count += 1
                #trial["state"] = hyperopt.JOB_STATE_RUNNING

                # elif trial["state"] == hyperopt.JOB_STATE_RUNNING:
                # check if all results files are there and set to ok
                hyperparam = self.unwrap_hyperparam(trial["misc"]["vals"])
                full_path = self.make_full_path(hyperparam)
                finished_ids = rt.get_finished_ids(path=full_path)
                if set(finished_ids).issuperset(set(self.ids)):
                    trial["state"] = hyperopt.JOB_STATE_DONE
                    print trial["tid"], "done"
                    trial["result"] = self.get_results(full_path)
                    print "Parameters", hyperparam
        return count

    def get_results(self, path):
        # all jobs should be done
        res = tres.load_results(path)
        mapping = {'max_steps': (-1., 'steps'), 'min_steps': (1., 'steps'),
                   'max_reward': (-1., 'return')}
        neg, quan = mapping[self.objective]
        avg, std, n_trials = tres.avg_quantity(res, quan)
        avg *= neg
        weights = (np.arange(len(avg)) + 1) ** 2
        loss = (avg * weights).sum() / weights.sum()
        print time.ctime()
        print "Loss: {:.4g}".format(loss)
        # use #steps/eps at the moment
        return {"loss": loss,
                "num_trials": n_trials[-1],
                "status": hyperopt.STATUS_OK,
                "std_last_mean": std[-1]}


def import_param_space(filename):
    """
    gets the variable param_space from a file without executing its __main__ section
    """
    content = ""
    with open(filename) as f:
        lines = f.readlines()
        for l in lines:
            if "if __name__ ==" in l:
                # beware: we assume that the __main__ execution block is the
                # last one in the file
                break
            content += l
    vars = {}
    exec(content, vars)
    return vars["param_space"]


def find_hyperparameters(
        setting, path, space=None, max_evals=100, trials_per_point=30,
        parallelization="sequential",
        objective="max_reward", max_concurrent_jobs=100):
    """
    This function does hyperparameter optimization for RLPy experiments with the
    hyperopt library.
    At the end an instance of the optimization trials is stored in "path"/trials.pck

    :param setting: file specifying the experimental setup.
        It contains a make_experiment function and a dictionary
        named param_space if the argument space is not used.
        For each key of param_space there needs to be an optional
        argument in make_experiment
    :param path: directory used to store all intermediate results.
    :param space: (optional) an alternative specification of the hyperparameter
        space
    :param max_evals: maximum number of evaluations of a single hyperparameter
        setting
    :param trials_per_point: specifies the number of independent runs (with
        different seeds) of the experiment for evaluating a single hyperparameter
        setting.
    :param parallelization: either **sequential**, **joblib**, **condor_all**
        or **condor_full**, **condor**.
        the condor options can be used in a computing cluster with a HTCondor
        machine. The joblib option parallelizes runs on one machine and sequential
        runs every experiment in sequence.
    :param objective: (optional) string specifying the objective to optimize,
        possible values are *max_reward*, *min_steps*, *max_steps*
    :param max_concurrent_jobs: only relevant for condor_full parallelization.
        specifies the maximum number of jobs that should run at the same time.
    :return: a tuple containing the best hyperarameter settings and the hyperopt
        trials instance of the optimization procedure
    """
    if space is None:
        space = import_param_space(setting)

    def f(hyperparam):
        """function to optimize by hyperopt"""

        # "temporary" directory to use
        full_path = os.path.join(
            path,
            "-".join([str(v) for v in hyperparam.values()]))

        # execute experiment
        rt.run(setting, location=full_path, ids=range(1, trials_per_point + 1),
               parallelization=parallelization, force_rerun=False, block=True, **hyperparam)

        # all jobs should be done
        res = tres.load_results(full_path)

        if objective == "max_steps":
            m, s, n = tres.avg_quantity(res, "steps")
            val = -m
            std = s[-1]
        elif objective == "min_steps":
            m, s, n = tres.avg_quantity(res, "steps")
            val = m
            std = s[-1]
        elif objective == "max_reward":
            m, s, n = tres.avg_quantity(res, "return")
            val = -m
            std = s[-1]
        else:
            print "unknown objective"
        weights = (np.arange(len(val)) + 1) ** 2
        loss = (val * weights).sum() / weights.sum()
        print time.ctime()
        print "Parameters", hyperparam
        print "Loss", loss
        # use #steps/eps at the moment
        return {"loss": loss,
                "num_trials": n[-1],
                "status": hyperopt.STATUS_OK,
                "std_last_mean": std}

    if parallelization == "condor_all":
        trials = CondorTrials(path=path, ids=range(1, trials_per_point + 1),
                              setting=setting, objective=objective)
        domain = hyperopt.Domain(dummy_f, space, rseed=123)
        rval = hyperopt.FMinIter(hyperopt.rand.suggest, domain, trials,
                                 max_evals=30,
                                 max_queue_len=30)
        rval.exhaust()
        rval = hyperopt.FMinIter(hyperopt.tpe.suggest, domain, trials,
                                 max_evals=max_evals,
                                 max_queue_len=1)
        rval.exhaust()
        best = trials.argmin
    elif parallelization == "condor_full":
        trials = _search_condor_parallel(path=path, setting=setting,
                                         objective=objective,
                                         space=space, max_evals=max_evals,
                                         trials_per_point=trials_per_point)
        best = trials.argmin
    else:
        trials = hyperopt.Trials()
        best = hyperopt.fmin(f, space=space, algo=hyperopt.tpe.suggest,
                             max_evals=max_evals, trials=trials)

    with open(os.path.join(path, 'trials.pck'), 'w') as f:
        pickle.dump(trials, f)

    return best, trials
