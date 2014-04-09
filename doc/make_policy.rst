.. _make_policy:

.. this is a comment. see http://sphinx-doc.org/rest.html for markup instructions

Creating a New Policy
=====================

This tutorial describes the standard RLPy 
:class:`~Policies.Policy.Policy` interface,
and illustrates a brief example of creating a new problem domain.

.. Below taken directly from Policy.py

The Policy determines the discrete action that an
:py:class:`~Agents.Agent.Agent` will take  given its current value function
:py:class:`~Representations.Representation.Representation`.

The Agent learns about the :py:class:`~Domains.Domain.Domain`
as the two interact.
At each step, the Agent passes information about its current state
to the Policy; the Policy uses this to decide what discrete action the
Agent should perform next (see :py:meth:`~Policies.Policy.Policy.pi`) \n


.. warning::
    While each dimension of the state *s* is either *continuous* or *discrete*,
    discrete dimensions are assume to take nonnegative **integer** values 
    (ie, the index of the discrete state).
        
.. note ::
    You may want to review the namespace / inheritance / scoping 
    `rules in Python <https://docs.python.org/2/tutorial/classes.html>`_.


Requirements 
------------

* At the top of the file (before the class definition), include the heading::

    __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
    __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann", 
                    "William Dabney", "Jonathan P. How"]
    __license__ = "BSD 3-Clause"
    __author__ = "Tim Beaver"

Fill in the appropriate ``__author__`` name and ``__credits__`` as needed.
Note that RLPy requires the BSD 3-Clause license.

* If available, please include a link or reference to the publication associated 
  with this implementation (and note differences, if any).

* Each Policy must be a subclass of :class:`~Policies.Policy.Policy` and call 
  the :func:`~Policies.Policy.__init__` function of the 
  Policy superclass.

* Accordingly, each Policy must be instantiated with a Logger (or None)
  in the ``__init__()`` function. Your code should be appropriately handle 
  the case where ``logger=None`` is passed to ``__init__()``.

* Once completed, the className of the new agent must be added to the
  ``__init__.py`` file in the ``Policies/`` directory.
  (This allows other files to import the new Policy).

* After your Policy is complete, you should define a unit test XX Add info here XX

REQUIRED Instance Variables
"""""""""""""""""""""""""""
---

REQUIRED Functions
""""""""""""""""""
#. :py:meth:`~Policies.Policy.Policy.pi` - accepts the current state *s*,
   whether or not *s* is *terminal*, and an array of possible actions 
   indices *p_actions* and returns an action index for the Agent to take.


SPECIAL Functions
"""""""""""""""""
Policies which have an explicit exploratory component (eg epsilon-greedy)
**MUST** override the functions below to prevent exploratory behavior
when evaluating the policy (which would skew results)

#. :py:meth:`~Policies.Policy.Policy.turnOffExploration`
#. :py:meth:`~Policies.Policy.Policy.turnOnExploration`


Additional Information
----------------------

* As always, the Policy can log messages using ``self.logger.log(<str>)``, see 
  :func:`Tools.Logger.log`. 
  Your code should be appropriately handle the case where ``logger=None`` is 
  passed to ``__init__()``.

* You should log values assigned to custom parameters when ``__init__()`` is called.

* See :class:`~Policies.Policy.Policy` for functions 
  provided by the superclass, especially before defining 
  helper functions which might be redundant. \n

* Note the useful functions provided by 
  the :class:`~Representations.Representation.Representation``,
  e.g. :func:`~Representations.Representation.bestActions` 
  and :func:`~Representations.Representation.bestAction`
  to get the best action(s) with respect to the value function (greedy).



Example: Creating the ``Epsilon-Greedy`` Policy
-----------------------------------------------------------
In this example we will recreate the ``eGreedy`` Policy.
From a given state, it selects the action with the highest expected value
(greedy with respect to value function), but with some probability ``epsilon``,
takes a random action instead.  This explicitly balances the exploration/exploitation
tradeoff, and ensures that in the limit of infinite samples, the agent will
have explored the entire domain.

#. Create a new file in the ``Policies/`` directory, ``eGreedyTut.py``.
   Add the header block at the top::

       __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
       __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                      "William Dabney", "Jonathan P. How"]
       __license__ = "BSD 3-Clause"
       __author__ = "Ray N. Forcement"
       
        from Policy import *
        import numpy as np

#. Declare the class, create needed members variables, and write a 
   docstring description.  See the role of member variables in comments::

       class eGreedy(Policy):
           """
           From the tutorial in policy creation.  Identical to eGreedy.py.
           """

           # Probability of selecting a random action instead of greedy
           epsilon         = None
           # Temporarily stores value of ``epsilon`` when exploration disabled
           old_epsilon     = None 
           # bool, used to avoid random selection among actions with the same values
           forcedDeterministicAmongBestActions = None

#. Copy the ``__init__()`` declaration from ``Policy.py`` and add needed parameters. 
   In the function body, assign them and log them.
   Then call the superclass constructor.
   Here the parameters are the probability of 
   selecting a random action, ``epsilon``, and how to handle the case where 
   multiple best actions exist, ie with the same 
   value, ``forcedDeterministicAmongBestActions``::

       def __init__(self,representation,logger,epsilon = .1,
                     forcedDeterministicAmongBestActions = False):
           self.epsilon = epsilon
           self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
           super(eGreedy,self).__init__(representation,logger)
           if self.logger:
               self.logger.log("=" * 60)
               self.logger.log("Policy: eGreedy")
               self.logger.log("Epsilon\t\t{0}".format(self.epsilon))


#. Copy the ``pi()`` declaration from ``Policy.py`` and implement it to return
   an action index for any given state and possible action inputs.
   Here, with probability epsilon, take a random action among the possible.
   Otherwise, pick an action with the highest expected value (depending on
   ``self.forcedDeterministicAmongBestActions``, either pick randomly from among
   the best actions or always select the one with lowest index::

       def pi(self,s, terminal, p_actions):
           coin = np.random.rand()
           #print "coin=",coin
           if coin < self.epsilon:
               return np.random.choice(p_actions)
           else:
               b_actions = self.representation.bestActions(s, terminal, p_actions)
               if self.forcedDeterministicAmongBestActions:
                   return b_actions[0]
               else:
                   return np.random.choice(b_actions)

#. Because this policy has an exploratory component, we must override the
   ``turnOffExploration()`` and ``turnOnExploration()`` functions, so that when
   evaluating the policy's performance the exploratory component may be
   automatically disabled so as not to influence results::

       def turnOffExploration(self):
           self.old_epsilon = self.epsilon
           self.epsilon = 0
       def turnOnExploration(self):
           self.epsilon = self.old_epsilon


.. warning::

    If you fail to define ``turnOffExploration()`` and ``turnOnExploration()``
    for functions with exploratory components, measured algorithm performance
    will be worse, since exploratory actions by definition are suboptimal based
    on the current model.

That's it! Now add your new Policy to ``Policies/__init__.py``::

    ``from eGreedyTut import eGreedyTut``

Finally, create a unit test for your Policy XX XX.

Now test it by creating a simple settings file on the domain of your choice.
An example experiment is given below:

.. literalinclude:: ../examples/tutorial/eGreedyTut_example.py
   :language: python
   :linenos:

What to do next?
----------------

In this Policy tutorial, we have seen how to 

* Write a Policy that inherits from the RLPy base ``Policy`` class
* Override several base functions, including those that manage exploration/exploitation
* Add the Policy to RLPy and test it

If you would like to add your new Policy to the RLPy project, email ``rlpy@mit.edu``
or create a pull request to the 
`RLPy repository <https://bitbucket.org/rlpy/rlpy>`_.

