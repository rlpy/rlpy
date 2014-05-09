from rlpy.Tools.hypersearch import find_hyperparameters
best, trials = find_hyperparameters(
    "examples/tutorial/infTrackCartPole_rbfs.py",
    "./Results/Tutorial/InfTrackCartPole/RBFs_hypersearch",
    max_evals=10, parallelization="joblib",
    trials_per_point=5)
print best
