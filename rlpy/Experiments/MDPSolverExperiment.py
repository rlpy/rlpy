"""Standard Experiment for Learning Control in RL"""

import rlpy.Tools.ipshell
from .Experiment import Experiment

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class MDPSolverExperiment(Experiment):

    """
    The MDPSolver Experiment connects an MDPSolver and Domain, and runs the MDPSolver's
    solve method to start solving the MDP.
    """

    #: The domain to be tested on
    domain = None
    #: The agent to be tested
    agent = None

    def __init__(self, agent, domain, **kwargs):
        self.agent = agent
        self.domain = domain

    def run(self, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

        debug_on_sigurg (boolean):
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """
        if debug_on_sigurg:
            rlpy.Tools.ipshell.ipdb_on_SIGURG()

        self.agent.solve()
