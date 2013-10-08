from Tools.hypersearch import find_hyperparameters
best, trials = find_hyperparameters("examples/tutorial/pendulum_rbfs.py",
                                    "./Results/Tutorial/Pendulum/RBFs_hypersearch",
                                    max_evals=10, parallelization="joblib",
                                    trials_per_point=5)
print best
