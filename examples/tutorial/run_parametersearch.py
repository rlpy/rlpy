from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from rlpy.Tools.hypersearch import find_hyperparameters
best, trials = find_hyperparameters(
    "examples/tutorial/infTrackCartPole_rbfs.py",
    "./Results/Tutorial/InfTrackCartPole/RBFs_hypersearch",
    max_evals=10, parallelization="joblib",
    trials_per_point=5)
print(best)
