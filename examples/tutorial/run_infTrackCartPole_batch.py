from builtins import range
from rlpy.Tools.run import run
run("examples/tutorial/infTrackCartPole_rbfs.py", "./Results/Tutorial/InfTrackCartPole/RBFs",
    ids=list(range(10)), parallelization="joblib")

run("examples/tutorial/infTrackCartPole_tabular.py", "./Results/Tutorial/InfTrackCartPole/Tabular",
    ids=list(range(10)), parallelization="joblib")
