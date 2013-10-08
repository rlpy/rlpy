from Tools.run import run
run("examples/tutorial/pendulum_rbfs.py", "./Results/Tutorial/Pendulum/RBFs",
    ids=range(10), parallelization="joblib")

run("examples/tutorial/pendulum_tabular.py", "./Results/Tutorial/Pendulum/Tabular",
    ids=range(10), parallelization="joblib")
