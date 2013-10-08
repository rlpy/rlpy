import Tools.results as rt

paths = {"RBFs": "./Results/Tutorial/Pendulum/RBFs",
         "Tabular": "./Results/Tutorial/Pendulum/Tabular"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps", "return")
rt.save_figure(fig, "./Results/Tutorial/plot.pdf")
