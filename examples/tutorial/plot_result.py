import rlpy.Tools.results as rt

paths = {"RBFs": "./Results/Tutorial/InfTrackCartPole/RBFs",
         "Tabular": "./Results/Tutorial/InfTrackCartPole/Tabular"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps", "return")
rt.save_figure(fig, "./Results/Tutorial/plot.pdf")
