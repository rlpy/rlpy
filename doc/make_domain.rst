.. _make_domain:

.. this is a comment. see http://sphinx-doc.org/rest.html for markup instructions

Creating a New Domain
=====================

This tutorial describes the standard RLPy 
:class:`~rlpy.Domains.Domain.Domain` interface,
and illustrates a brief example of creating a new problem domain.

.. Below taken directly from Domain.py

The Domain controls the environment in which the
:class:`~rlpy.Agents.Agent.Agent` resides as well as the reward function the
Agent is subject to.

The Agent interacts with the Domain in discrete timesteps called
*episodes* (see :func:`~rlpy.Domains.Domain.Domain.step`).
At each step, the Agent informs the Domain what indexed action it wants to
perform.  The Domain then calculates the effects this action has on the
environment and updates its internal state accordingly.
It also returns the new state (*ns*) to the agent, along with a reward/penalty, (*r*)
and whether or not the episode is over (*terminal*), in which case the agent
is reset to its initial state.

This process repeats until the Domain determines that the Agent has either
completed its goal or failed.
The :py:class:`~rlpy.Experiments.Experiment.Experiment` controls this cycle.

Because Agents are designed to be agnostic to the Domain that they are
acting within and the problem they are trying to solve, the Domain needs
to completely describe everything related to the task. Therefore, the
Domain must not only define the observations that the Agent receives,
but also the states it can be in, the actions that it can perform, and the
relationships between the three.

.. warning::
    While each dimension of the state *s* is either *continuous* or *discrete*,
    discrete dimensions are assume to take nonnegative **integer** values 
    (ie, the index of the discrete state).
        
.. note ::
    You may want to review the namespace / inheritance / scoping 
    `rules in Python <https://docs.python.org/2/tutorial/classes.html>`_.


Requirements 
------------

* Each Domain must be a subclass of 
  :class:`~rlpy.Domains.Domain.Domain` and call the 
  :func:`~rlpy.Domains.Domain.Domain.__init__` function of the 
  Domain superclass.

* Any randomization that occurs at object construction *MUST* occur in 
  the :func:`~rlpy.Domains.Domain.Domain.init_randomization` function, 
  which can be called by ``__init__()``.

* Any random calls should use ``self.random_state``, not ``random()`` or
  ``np.random()``, as this will ensure consistent seeded results during 
  experiments.

* After your agent is complete, you should define a unit test to ensure future
  revisions do not alter behavior.  See rlpy/tests/test_domains for some examples.


REQUIRED Instance Variables
"""""""""""""""""""""""""""
The new Domain *MUST* set these variables *BEFORE* calling the
superclass ``__init__()`` function:

#. ``self.statespace_limits`` - Bounds on each dimension of the state space. 
   Each row corresponds to one dimension and has two elements [min, max].
   Used for discretization of continuous dimensions.

#. ``self.continuous_dims`` - array of integers; each element is the index 
   (eg, row in ``statespace_limits`` above) of a continuous-valued dimension.
   This array is empty if all states are discrete.

#. ``self.DimNames`` - array of strings, a name corresponding to each dimension
   (eg one for each row in ``statespace_limits`` above)

#. ``self.episodeCap`` - integer, maximum number of steps before an episode
   terminated (even if not in a terminal state).

#. ``actions_num`` - integer, the total number of possible actions (ie, the size
   of the action space).  This number **MUST** be a finite integer - continuous action
   spaces are not currently supported.

#. ``discount_factor`` - float, the discount factor (gamma in literature)
   by which rewards are reduced.


REQUIRED Functions
""""""""""""""""""
#. :func:`~rlpy.Domains.Domain.Domain.s0`,
   (see linked documentation), which returns a (possibly random) state in the 
   domain, to be used at the start of an *episode*.

#. :func:`~rlpy.Domains.Domain.Domain.step`,
   (see linked documentation), which returns the tuple ``(r,ns,terminal, pa)`` 
   that results from taking action *a* from the current state (internal to the Domain).

   * *r* is the reward obtained during the transition
   * *ns* is the new state after the transition
   * *terminal*, a boolean, is true if the new state *ns* is a terminal one to end the episode
   * *pa*, an array of possible actions to take from the new state *ns*.


SPECIAL Functions
"""""""""""""""""
In many cases, the Domain will also override the functions:

#. :func:`~rlpy.Domains.Domain.Domain.isTerminal` - returns a boolean whether or
   not the current (internal) state is terminal. Default is always return False.
#. :func:`~rlpy.Domains.Domain.Domain.possibleActions` - returns an array of
   possible action indices, which often depend on the current state.
   Default is to enumerate **every** possible action, regardless of current state.


OPTIONAL Functions
""""""""""""""""""
Optionally, define / override the following functions, used for visualization:

#. :func:`~rlpy.Domains.Domain.Domain.showDomain` - Visualization of domain based
   on current internal state and an action, *a*.
   Often the header will include an optional argument *s* to display instead 
   of the current internal state.
   RLPy frequently uses `matplotlib <http://matplotlib.org/>`_
   to accomplish this - see the example below.
#. :func:`~rlpy.Domains.Domain.Domain.showLearning` - Visualization of the "learning"
   obtained so far on this domain, usually a value function plot and policy plot.
   See the introductory tutorial for an example on :class:`~rlpy.Domains.Gridworld.GridWorld`

XX expectedStep(), XX


Additional Information
----------------------

* As always, the Domain can log messages using ``self.logger.info(<str>)``, see 
  Python ``logger`` doc.

* You should log values assigned to custom parameters when ``__init__()`` is called.

* See :class:`~rlpy.Domains.Domain.Domain` for functions 
  provided by the superclass, especially before defining 
  helper functions which might be redundant.


Example: Creating the ``ChainMDP`` Domain
-----------------------------------------------------------
In this example we will recreate the simple ``ChainMDP`` Domain, which consists
of *n* states that can only transition to *n-1* or *n+1*:
``s0 <-> s1 <-> ... <-> sn`` \n
The goal is to reach state ``sn`` from ``s0``, after which the episode terminates.
The agent can select from two actions: left [0] and right [1] (it never remains in same state).
But the transitions are noisy, and the opposite of the desired action is taken 
instead with some probability.
Note that the optimal policy is to always go right.

#. Create a new file in your current working directory, ``ChainMDPTut.py``.
   Add the header block at the top::

        __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
        __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                       "William Dabney", "Jonathan P. How"]
        __license__ = "BSD 3-Clause"
        __author__ = "Ray N. Forcement"

        from rlpy.Tools import plt, mpatches, fromAtoB
        from rlpy.Domains.Domain import Domain
        import numpy as np

#. Declare the class, create needed members variables (here several objects to
   be used for visualization and a few domain reward parameters), and write a 
   docstring description::

        class ChainMDPTut(Domain):
            """
            Tutorial Domain - nearly identical to ChainMDP.py
            """
            #: Reward for each timestep spent in the goal region
            GOAL_REWARD = 0
            #: Reward for each timestep
            STEP_REWARD = -1
            #: Set by the domain = min(100,rows*cols)
            episodeCap = 0
            # Used for graphical normalization
            MAX_RETURN  = 1
            # Used for graphical normalization
            MIN_RETURN  = 0
            # Used for graphical shifting of arrows
            SHIFT       = .3
            #:Used for graphical radius of states
            RADIUS      = .5
            # Stores the graphical pathes for states so that we can later change their colors
            circles     = None
            #: Number of states in the chain
            chainSize   = 0
            # Y values used for drawing circles
            Y           = 1

#. Copy the __init__ declaration from ``Domain.py``, add needed parameters
   (here the number of states in the chain, ``chainSize``), and log them.
   Assign ``self.statespace_limits, self.episodeCap, self.continuous_dims, self.DimNames, self.actions_num,`` 
   and ``self.discount_factor``.
   Then call the superclass constructor::

            def __init__(self, chainSize=2):
                """
                :param chainSize: Number of states \'n\' in the chain.
                """
                self.chainSize          = chainSize
                self.start              = 0
                self.goal               = chainSize - 1
                self.statespace_limits  = np.array([[0,chainSize-1]])
                self.episodeCap         = 2*chainSize
                self.continuous_dims    = []
                self.DimNames           = ['State']
                self.actions_num        = 2
                self.discount_factor    = 0.9
                super(ChainMDPTut,self).__init__()

#. Copy the ``step()`` and function declaration and implement it accordingly
   to return the tuple (r,ns,isTerminal,possibleActions), and similarly for ``s0()``.
   We want the agent to always start at state *[0]* to begin, and only achieves reward 
   and terminates when *s = [n-1]*::

            def step(self,a):
                s = self.state[0]
                if a == 0: #left
                    ns = max(0,s-1)
                if a == 1: #right
                    ns = min(self.chainSize-1,s+1)
                self.state = np.array([ns])

                terminal = self.isTerminal()
                r = self.GOAL_REWARD if terminal else self.STEP_REWARD
                return r, ns, terminal, self.possibleActions()

            def s0(self):
                self.state = np.array([0])
                return self.state, self.isTerminal(), self.possibleActions()

#. In accordance with the above termination condition, override the ``isTerminal()``
   function by copying its declaration from ``Domain.py``::

            def isTerminal(self):
                s = self.state
                return (s[0] == self.chainSize - 1)

#. For debugging convenience, demonstration, and entertainment, create a domain
   visualization by overriding the default (which is to do nothing).
   With matplotlib, generally this involves first performing a check to see if
   the figure object needs to be created (and adding objects accordingly),
   otherwise merely updating existing plot objects based on the current ``self.state``
   and action *a*::


            def showDomain(self, a = 0):
                #Draw the environment
                s = self.state
                s = s[0]
                if self.circles is None: # We need to draw the figure for the first time
                   fig = plt.figure(1, (self.chainSize*2, 2))
                   ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
                   ax.set_xlim(0, self.chainSize*2)
                   ax.set_ylim(0, 2)
                   ax.add_patch(mpatches.Circle((1+2*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
                   ax.xaxis.set_visible(False)
                   ax.yaxis.set_visible(False)
                   self.circles = [mpatches.Circle((1+2*i, self.Y), self.RADIUS, fc="w") for i in np.arange(self.chainSize)]
                   for i in np.arange(self.chainSize):
                       ax.add_patch(self.circles[i])
                       if i != self.chainSize-1:
                            fromAtoB(1+2*i+self.SHIFT,self.Y+self.SHIFT,1+2*(i+1)-self.SHIFT, self.Y+self.SHIFT)
                            if i != self.chainSize-2: fromAtoB(1+2*(i+1)-self.SHIFT,self.Y-self.SHIFT,1+2*i+self.SHIFT, self.Y-self.SHIFT, 'r')
                       fromAtoB(.75,self.Y-1.5*self.SHIFT,.75,self.Y+1.5*self.SHIFT,'r',connectionstyle='arc3,rad=-1.2')
                       plt.show()

                [p.set_facecolor('w') for p in self.circles]
                self.circles[s].set_facecolor('k')
                plt.draw()

.. note::

    When first creating a matplotlib figure, you must call pl.show(); when
    updating the figure on subsequent steps, use pl.draw().

That's it!  Now test it by creating a simple settings file on the domain of your choice.
An example experiment is given below:

.. literalinclude:: ../examples/tutorial/ChainMDPTut_example.py
   :language: python
   :linenos:

What to do next?
----------------

In this Domain tutorial, we have seen how to 

* Write a Domain that inherits from the RLPy base ``Domain`` class
* Override several base functions
* Create a visualization
* Add the Domain to RLPy and test it

Adding your component to RLPy
"""""""""""""""""""""""""""""
If you would like to add your component to RLPy, we recommend developing on the 
development version (see :ref:`devInstall`).
Please use the following header template at the top of each file:: 

    __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
    __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann", 
                    "William Dabney", "Jonathan P. How"]
    __license__ = "BSD 3-Clause"
    __author__ = "Tim Beaver"

Fill in the appropriate ``__author__`` name and ``__credits__`` as needed.
Note that RLPy requires the BSD 3-Clause license.

* If you installed RLPy in a writeable directory, the className of the new 
  domain can be added to 
  the ``__init__.py`` file in the ``Domains/`` directory.
  (This allows other files to import the new domain).

* If available, please include a link or reference to the publication associated 
  with this implementation (and note differences, if any).

If you would like to add your new domain to the RLPy project, we recommend
you branch the project and create a pull request to the 
`RLPy repository <https://bitbucket.org/rlpy/rlpy>`_.

You can also email the community list ``rlpy@mit.edu`` for comments or 
questions. To subscribe `click here <http://mailman.mit.edu/mailman/listinfo/rlpy>`_.

