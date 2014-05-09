from rlpy.Tools.run import run
run("examples/tutorial/infTrackCartPole_rbfs.py", "./Results/Tutorial/InfTrackCartPole/RBFs",
    ids=range(10), parallelization="joblib")

run("examples/tutorial/infTrackCartPole_tabular.py", "./Results/Tutorial/InfTrackCartPole/Tabular",
    ids=range(10), parallelization="joblib")
