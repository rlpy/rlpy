from rlpy.Tools import plt
import numpy as np


def _checkSameExperimentResults(exp1, exp2):
    """ Returns False if experiments gave same results, true if they match. """
    if not np.all(exp1.result["learning_steps"] == exp2.result["learning_steps"]):
        # Same number of steps before failure (where applicable)
        print 'LEARNING STEPS DIFFERENT'
        print exp1.result["learning_steps"]
        print exp2.result["learning_steps"]
        return False
    if not np.all(exp1.result["return"] == exp2.result["return"]):
        # Same return on each test episode
        print 'RETURN DIFFERENT'
        print exp1.result["return"]
        print exp2.result["return"]
        return False
    if not np.all(exp1.result["steps"] == exp2.result["steps"]):
        # Same number of steps taken on each training episode
        print 'STEPS DIFFERENT'
        print exp1.result["steps"]
        print exp2.result["steps"]
        return False
    return True

def check_seed_vis(make_exp_fun):
    """ Ensure that providing the same random seed yields same result """
    # [[initialize and run experiment without visual]]
    expNoVis = make_exp_fun(exp_id=1)
    expNoVis.config_logging = False
    expNoVis.run(visualize_steps=False,
            visualize_learning=False,
            visualize_performance=0)

    # [[initialize and run experiment with visual]]
    expVis1 = make_exp_fun(exp_id=1)
    expVis1.config_logging = False
    expVis1.run(visualize_steps=True,
            visualize_learning=False,
            visualize_performance=1)
    plt.close('all')

    expVis2 = make_exp_fun(exp_id=1)
    expVis2.config_logging = False
    expVis2.run(visualize_steps=False,
            visualize_learning=True,
            visualize_performance=1)
    plt.close('all')

    # [[assert get same results]]
    assert _checkSameExperimentResults(expNoVis, expVis1)
    assert _checkSameExperimentResults(expNoVis, expVis2)


