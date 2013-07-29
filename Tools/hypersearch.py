import os
from Tools.Merger import Merger
import Tools.run as rt
import hyperopt
import numpy as np


def dummy_f():
    pass


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
        return os.path.join(self.path, "-".join([str(v) for v in hyperparam.values()]))

    def update_trials(self, trials):
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
                #trial["state"] = hyperopt.JOB_STATE_RUNNING

                #elif trial["state"] == hyperopt.JOB_STATE_RUNNING:
                # check if all results files are there and set to ok
                hyperparam = self.unwrap_hyperparam(trial["misc"]["vals"])
                full_path = self.make_full_path(hyperparam)
                finished_ids = rt.get_finished_ids(path=full_path)
                if set(finished_ids).issuperset(set(self.ids)):
                    trial["state"] = hyperopt.JOB_STATE_DONE
                    print trial["tid"], "done"
                    trial["result"] = self.get_results(full_path)

    def get_results(self, path):
        # all jobs should be done
        m = Merger([path], showSplash=False)
        if self.objective == "max_steps":
            val = -m.means[0][4, :]
            idx = 4
        elif self.objective == "min_steps":
            val = m.means[0][4, :]
            idx = 4
        elif self.objective == "max_reward":
            val = -m.means[0][1, :]
            idx = 1
        else:
            print "unknown objective"
        weights = (np.arange(len(val)) + 1) ** 2
        loss = (val * weights).sum() / weights.sum()
        print "Loss", loss
        # use #steps/eps at the moment
        return {"loss": loss,
                "num_trials": m.samples[0],
                "status": hyperopt.STATUS_OK,
                "std_last_mean": m.std_errs[0][idx, -1]}

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

def find_hyperparameters(setting, path, space=None, max_evals=100, trials_per_point=30,
                         parallelization="sequential",
                         objective="max_reward", max_concurrent_jobs=100):
    """
    This function does hyperparameter optimization for RLPy experiments with the
    hyperopt library.
    """
    if space is None:
        space = import_param_space(setting)

    def f(hyperparam):
        """function to optimize by hyperopt"""

        # "temporary" directory to use
        full_path = os.path.join(path, "-".join([str(v) for v in hyperparam.values()]))

        # execute experiment
        rt.run(setting, location=full_path, ids=range(1, trials_per_point + 1),
               parallelization=parallelization, force_rerun=False, block=True, **hyperparam)

        # all jobs should be done
        m = Merger([full_path], minSamples=trials_per_point, showSplash=False)
        if objective == "max_steps":
            val = -m.means[0][4, :]
            idx = 4
        elif objective == "min_steps":
            val = m.means[0][4, :]
            idx = 4
        elif objective == "max_reward":
            val = -m.means[0][1, :]
            idx = 1
        else:
            print "unknown objective"
        weights = (np.arange(len(val)) + 1) ** 2
        loss = (val * weights).sum() / weights.sum()
        print "Loss", loss
        # use #steps/eps at the moment
        return {"loss": loss,
                "num_trials": m.samples[0],
                "status": hyperopt.STATUS_OK,
                "std_last_mean": m.std_errs[0][idx, -1]}

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
    else:
        trials = hyperopt.Trials()
        best = hyperopt.fmin(f, space=space, algo=hyperopt.tpe.suggest,
                             max_evals=max_evals, trials=trials)

    return best, trials
