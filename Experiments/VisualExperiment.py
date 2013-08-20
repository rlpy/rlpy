from Experiment import Experiment
from time import clock
from Tools import deltaT, hhmmss

__author__ = "Christoph Dann"


class VisualExperiment(Experiment):
    """
    Experiment that just shows the value function
    and does not do any performance runs at all
    """

    def __init__(self, vis_learning_freq=1, **kwargs):
        self.vis_learning_freq = vis_learning_freq
        super(VisualExperiment, self).__init__(**kwargs)

    def save(self):
        pass

    def run(self, visualize_steps=False, **kwargs):
        """
        Run the online experiment and collect statistics
        """

        self.seed_components()
        terminal            = True
        total_steps         = 0
        eps_steps           = 0
        performance_tick    = 0
        eps_return          = 0
        episode_number      = 0
        start_log_time      = clock() # Used to bound the number of logs in the file
        self.start_time     = clock() # Used to show the total time took the process
        self.domain.showLearning(self.agent.representation)
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                s           = self.domain.s0()
                a           = self.agent.policy.pi(s)
                #Visual
                if visualize_steps: self.domain.show(s,a, self.agent.representation)
                # Hash new state for the tabular case
                # Output the current status if certain amount of time has been passed
                eps_return      = 0
                eps_steps       = 0
                episode_number += 1

            #Act,Learn,Step
            r,ns,terminal   = self.domain.step(s, a)
            na              = self.agent.policy.pi(ns)
            total_steps     += 1
            eps_steps       += 1
            eps_return      += r
            terminal |= eps_steps >= self.domain.episodeCap

            #Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap) and deltaT(start_log_time) > self.log_interval:
                start_log_time  = clock()
                elapsedTime     = deltaT(self.start_time)
                self.logger.log('%d: E[%s]-R[%s]: Return=%+0.2f, Steps=%d, Features = %d' % (total_steps, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps), eps_return, eps_steps, self.agent.representation.features_num))
            self.agent.learn(s,a,r,ns,na,terminal)
            s,a          = ns,na
            if self.vis_learning_freq and total_steps % self.vis_learning_freq == 0:
                self.agent.representation.plot_1d_features()
                self.agent.representation.plot_2d_feature_centers()
                self.domain.showLearning(self.agent.representation)

        if visualize_steps: self.domain.show(s,a, self.agent.representation)    