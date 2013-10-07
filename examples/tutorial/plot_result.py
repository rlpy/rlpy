import Tools.results as restools

paths = {"RBFs": "./Results/Tutorial/Pendulum/RBFs",
         "Tabular": "./Results/Tutorial/Pendulum/Tabular"}

merger = restools.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps", "return")
restools.save_figure(fig, "./Results/Tutorial/plot.pdf")
