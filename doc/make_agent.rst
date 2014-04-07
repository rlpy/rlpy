.. _make_agent:

.. this is a comment. see http://sphinx-doc.org/rest.html for markup instructions

Creating a New Agent
===============

This tutorial describes the standard RLPy :class:`Agents.Agent` interface,
and illustrates a brief example of creating a new learning agent.

.. Below taken directly from Agent.py

The Agent receives observations from the Domain and updates the 
Representation accordingly.

In a typical Experiment, the Agent interacts with the Domain in discrete 
timesteps.
At each Experiment timestep the Agent receives some observations from the Domain
which it uses to update the value function Representation of the Domain
(ie, on each call to its :py:meth:`~Agents.Agent.Agent.learn` function).
The Policy is used to select an action to perform.
This process (observe, update, act) repeats until some goal or fail state,
determined by the Domain, is reached. At this point the
:py:class:`~Experiments.Experiment.Experiment` determines
whether the agent starts over or has its current policy tested
(without any exploration).

.. note ::
    You may want to review the namespace / inheritance / scoping 
    .. _rules in Python: https://docs.python.org/2/tutorial/classes.html


Requirements 
---------

* At the top of the file (before the class definition), include the heading::
    __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
    __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                   "William Dabney", "Jonathan P. How"]
    __license__ = "BSD 3-Clause"
    __author__ = "Christoph Dann"

Fill in the appropriate ``__author__'' name and ``__credits__'' as needed.
Note that RLPy requires the BSD 3-Clause license.

* If available, please include a link or reference to the publication associated 
with this implementation (and note differences, if any).

* Each learning agent must be a subclass of :class:`Agents.Agent` and call the 
__init__() function of the Agent superclass.

* Accordingly, each Agent must be instantiated with a Logger (or None), Representation, 
Policy, and Domain XX Remove additional params eg boyan? XX in the 
:func:`Agents.Agent.__init__` function.

* Your code should be appropriately handle the case where ``logger=None'' is 
passed to ``__init__''.

* The new learning agent need only define the :func:`Agents.Agent.learn` function, (see
linked documentation) which is called on every timestep.
..Note:: 

    The Agent *MUST* call the (inherited) :func:`Agents.Agent.episodeTerminated'
    function after learning if the transition led to a terminal state
    (ie, learn() will return isTerminal=True)

..Note::
    The ``learn()'' function *MUST* call the :func:`Representations.Representation.pre_discover'
    function at its beginning, and :func:`Representations.Representation.post_discover'
    at its end.  This allows adaptive representations to add new features
    (no effect on fixed ones).

* Once completed, the className of the new agent must be added to the
``__init__.py'' file in the ``Agents/'' directory.
(This allows other files to import the new agent).

* After your agent is complete, you should define a unit test XX Add info here XX


Additional Information
-----------------------

* As always, the agent can log messages using ``self.logger.log(<str>)'', see 
:func:`Tools.Logger.log'. 
Your code should be appropriately handle the case where ``logger=None'' is 
passed to ``__init__''.

* You should write values assigned to custom parameters when __init__ is called.

* See :class:`Agents.Agent' for functions provided by the ``Agent'' superclass.



Example: Creating the ``SARSA0'' Agent
--------------------------------------------
In this example, we will create the standard SARSA learning agent (without 
eligibility traces (ie the 0xCE 0xBB parameter= 0 always)).
This algorithm first computes the Temporal Difference Error
(see, Sutton and Barto's *Reinforcement Learning* (1998) or 
.. _Wikipedia: http://en.wikipedia.org/wiki/Temporal_difference_learning),
essentially the difference between the prediction under the current 
value function and what was actually observed.
It then updates the representation by summing the current function with 
this TD error, weighted by a factor called the *learning rate*.


#. Create a new file in the ``Agents/'' directory, ``SARSA0.py''.
Add the header block at the top::
    __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
    __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                   "William Dabney", "Jonathan P. How"]
    __license__ = "BSD 3-Clause"
    __author__ = "Ray N. Forcement"

    from Agent import Agent
    import numpy

#. Declare the class, create needed members variables, and write a 
docstring description::
    class SARSA0(Agent):
        """
        Standard SARSA algorithm without eligibility trace (ie lambda=0)
        """
        learning_rate = 0 # The weight on TD updates ('alpha' in the paper)
#. Copy the __init__ declaration from ``Agent.py'', add the learning_rate
parameter, and log the passed value::
    def __init__(self, logger, representation, policy, domain, learning_rate=0.1):
        self.learning_rate = learning_rate
        super(SARSA0,self).__init__(representation,policy,domain,logger,initial_alpha,alpha_decay_mode, boyan_N0)
        if logger:
            self.logger.log("Learning rate:\t\t%0.2f" % learning_rate)

#. Copy the learn() declaration, compute the td-error, and use it to update
the value function estimate (by adjusting feature weights)::
   def learn(self,s,p_actions, a, r, ns, np_actions, na,terminal):

        # The previous state could never be terminal
        # (otherwise the episode would have already terminated)
        prevStateTerminal = False 

        # MUST call this at start of learn()
        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

        # Compute feature function values and next action to be taken

        discount_factor = self.representation.domain.gamma # 'gamma' in literature
        feat_weights    = self.representation.theta # Value function, expressed as feature weights
        features_s      = self.representation.phi(s, prevStateTerminal) # active feats in state
        features        = self.representation.phi_sa(s, prevStateTerminal, a, features_s) # active features for an (s,a) pair
        features_prime_s= self.representation.phi(ns, terminal)
        features_prime  = self.representation.phi_sa(ns, terminal, na, features_prime_s)
        nnz             = count_nonzero(phi_s)    # Number of non-zero elements

        # Compute td-error
        td_error            = r + np.dot(discount_factor*features_prime - features, theta)

        # Update value function (or if TD-learning diverges, take no action)
        if nnz > 0:
            feat_weights_old = feat_weights.copy()
            feat_weights               += self.alpha * td_error
            if not np.all(np.isfinite(theta)):
                feat_weights = feat_weights_old
                print "WARNING: TD-Learning diverged, theta reached infinity!"

        # MUST call this at end of learn() - add new features to representation as required.
        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)

        # MUST call this at end of learn() - handle episode termination cleanup as required.
        if terminal:
            self.episodeTerminated()



That's it! Now add your new agent to ``Agents/__init__.py'': 
``from SARSA0 import SARSA0''

Finally, create a unit test for your agent XX XX.

Now test it by creating a simple settings file on the domain of your choice.
An example experiment is given below:

.. literalinclude:: ../examples/tutorial/SARSA0_example.py
   :language: python
   :linenos:


This is a subsection
^^^^^^^^^^^^^^^^^^^^

What to do next?
----------------

In this Agent tutorial, we have seen how to 

* Write a learning agent that inherits from the RLPy base ``Agent'' class
* Add the agent to RLPy and test it

If you would like to add your new agent to the RLPy project, email ``rlpy@mit.edu''
or create a pull request to the 
.. _rlpy repository`https://bitbucket.org/rlpy/rlpy'.


.. epigraph::
    
    The only real mistake is the one from which we learn nothing.
    
    -- John Powell
