.. _make_agent:

.. this is a comment. see http://sphinx-doc.org/rest.html for markup instructions

Creating a New Agent
====================

This tutorial describes the standard RLPy :class:`~rlpy.Agents.Agent.Agent` interface,
and illustrates a brief example of creating a new learning agent.

.. Below taken directly from Agent.py

The Agent receives observations from the Domain and updates the 
Representation accordingly.

In a typical Experiment, the Agent interacts with the Domain in discrete 
timesteps.
At each Experiment timestep the Agent receives some observations from the Domain
which it uses to update the value function Representation of the Domain
(ie, on each call to its :func:`~rlpy.Agents.Agent.Agent.learn` function).
The Policy is used to select an action to perform.
This process (observe, update, act) repeats until some goal or fail state,
determined by the Domain, is reached. At this point the
Experiment determines
whether the agent starts over or has its current policy tested
(without any exploration).

.. note ::
    You may want to review the namespace / inheritance / scoping 
    `rules in Python <https://docs.python.org/2/tutorial/classes.html>`_.


Requirements 
------------

* Each learning agent must be a subclass of :class:`~rlpy.Agents.Agent.Agent` 
  and call 
  the :func:`~rlpy.Agents.Agent.Agent.__init__` function of the Agent superclass.

* Accordingly, each Agent must be instantiated with a Representation, 
  Policy, and Domain in the ``__init__()`` function

* Any randomization that occurs at object construction *MUST* occur in
  the :func:`~rlpy.Agents.Agent.Agent.init_randomization` function, 
  which can be called by ``__init__()``.

* Any random calls should use ``self.random_state``, not ``random()`` or 
  ``np.random()``, as this will ensure consistent seeded results during 
  experiments.

* After your agent is complete, you should define a unit test to ensure future
  revisions do not alter behavior.  See rlpy/tests for some examples.

REQUIRED Instance Variables
"""""""""""""""""""""""""""
---

REQUIRED Functions
""""""""""""""""""
:func:`~rlpy.Agents.Agent.Agent.learn` - called on every timestep (see documentation)

  .. Note:: 

      The Agent *MUST* call the (inherited) :func:`~rlpy.Agents.Agent.Agent.episodeTerminated`
      function after learning if the transition led to a terminal state
      (ie, ``learn()`` will return ``isTerminal=True``)

  .. Note::

      The ``learn()`` function *MUST* call the 
      :func:`~rlpy.Representations.Representation.Representation.pre_discover`
      function at its beginning, and 
      :func:`~rlpy.Representations.Representation.Representation.post_discover`
      at its end.  This allows adaptive representations to add new features
      (no effect on fixed ones).


Additional Information
----------------------

* As always, the agent can log messages using ``self.logger.info(<str>)``, see 
  the Python ``logger`` documentation

* You should log values assigned to custom parameters when ``__init__()`` is called.

* See :class:`~rlpy.Agents.Agent.Agent` for functions provided by the superclass.



Example: Creating the ``SARSA0`` Agent
--------------------------------------
In this example, we will create the standard SARSA learning agent (without 
eligibility traces (ie the λ parameter= 0 always)).
This algorithm first computes the Temporal Difference Error,
essentially the difference between the prediction under the current 
value function and what was actually observed
(see e.g. `Sutton and Barto's *Reinforcement Learning* (1998) <http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node60.html>`_ 
or `Wikipedia <http://en.wikipedia.org/wiki/Temporal_difference_learning>`_).
It then updates the representation by summing the current function with 
this TD error, weighted by a factor called the *learning rate*.


#. Create a new file in the current working directory, ``SARSA0.py``.
   Add the header block at the top::

        __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
        __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                       "William Dabney", "Jonathan P. How"]
        __license__ = "BSD 3-Clause"
        __author__ = "Ray N. Forcement"

        from rlpy.Agents.Agent import Agent, DescentAlgorithm
        import numpy

#. Declare the class, create needed members variables (here a learning rate),
   described above) and write a docstring description::

        class SARSA0(DescentAlgorithm, Agent):
            """
            Standard SARSA algorithm without eligibility trace (ie lambda=0)
            """

#. Copy the __init__ declaration from ``Agent`` and ``DescentAlgorithm``
   in ``Agent.py``, and add needed parameters
   (here the initial_learn_rate) and log them.  (kwargs is a catch-all for
   initialization parameters.)  Then call the superclass constructor::

            def __init__(self, policy, representation, discount_factor, initial_learn_rate=0.1, **kwargs):
                super(SARSA0,self).__init__(policy=policy,
                 representation=representation, discount_factor=discount_factor, **kwargs)
                self.logger.info("Initial learning rate:\t\t%0.2f" % initial_learn_rate)

#. Copy the learn() declaration and implement accordingly.
   Here, compute the td-error, and use it to update
   the value function estimate (by adjusting feature weights)::

            def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

                # The previous state could never be terminal
                # (otherwise the episode would have already terminated)
                prevStateTerminal = False

                # MUST call this at start of learn()
                self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

                # Compute feature function values and next action to be taken

                discount_factor = self.discount_factor # 'gamma' in literature
                feat_weights    = self.representation.weight_vec # Value function, expressed as feature weights
                features_s      = self.representation.phi(s, prevStateTerminal) # active feats in state
                features        = self.representation.phi_sa(s, prevStateTerminal, a, features_s) # active features or an (s,a) pair
                features_prime_s= self.representation.phi(ns, terminal)
                features_prime  = self.representation.phi_sa(ns, terminal, na, features_prime_s)
                nnz             = count_nonzero(features_s)  # Number of non-zero elements

                # Compute td-error
                td_error            = r + np.dot(discount_factor * features_prime - features, feat_weights)

                # Update value function (or if TD-learning diverges, take no action)
                if nnz > 0:
                    feat_weights_old = feat_weights.copy()
                    feat_weights               += self.learn_rate * td_error
                    if not np.all(np.isfinite(feat_weights)):
                        feat_weights = feat_weights_old
                        print "WARNING: TD-Learning diverged, theta reached infinity!"

                # MUST call this at end of learn() - add new features to representation as required.
                expanded = self.representation.post_discover(s, False, a, td_error, features_s)

                # MUST call this at end of learn() - handle episode termination cleanup as required.
                if terminal:
                    self.episodeTerminated()

.. note::

    You can and should define helper functions in your agents as needed, and 
    arrange class hierarchy. (See eg TDControlAgent.py)


That's it! Now test the agent by creating a simple settings file on the domain of your choice.
An example experiment is given below:

.. literalinclude:: ../examples/tutorial/SARSA0_example.py
   :language: python
   :linenos:

What to do next?
----------------

In this Agent tutorial, we have seen how to 

* Write a learning agent that inherits from the RLPy base ``Agent`` class
* Add the agent to RLPy and test it


Adding your component to RLPy
"""""""""""""""""""""""""""""
If you would like to add your component to RLPy, we recommend developing on the 
development version (see :ref:`devInstall`).
Please use the following header at the top of each file:: 

    __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
    __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann", 
                    "William Dabney", "Jonathan P. How"]
    __license__ = "BSD 3-Clause"
    __author__ = "Tim Beaver"

* Fill in the appropriate ``__author__`` name and ``__credits__`` as needed.
  Note that RLPy requires the BSD 3-Clause license.

* If you installed RLPy in a writeable directory, the className of the new 
  agent can be added to
  the ``__init__.py`` file in the ``Agents/`` directory.
  (This allows other files to import the new agent).

* If available, please include a link or reference to the publication associated 
  with this implementation (and note differences, if any).

If you would like to add your new agent to the RLPy project, we recommend
you branch the project and create a pull request to the 
`RLPy repository <https://bitbucket.org/rlpy/rlpy>`_.

You can also email the community list ``rlpy@mit.edu`` for comments or 
questions. To subscribe `click here <http://mailman.mit.edu/mailman/listinfo/rlpy>`_.

