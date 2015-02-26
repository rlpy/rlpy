.. _make_rep:

.. this is a comment. see http://sphinx-doc.org/rest.html for markup instructions

Creating a New Representation
=============================

This tutorial describes the standard RLPy 
:class:`~rlpy.Representations.Representation.Representation` interface,
and illustrates a brief example of creating a new value function representation.

.. Below taken directly from Representation.py

The Representation is the approximation of the
value function associated with a :py:class:`~rlpy.Domains.Domain.Domain`,
usually in some lower-dimensional feature space.

The Agent receives observations from the Domain on each step and calls 
its :func:`~rlpy.Agents.Agent.Agent.learn` function, which is responsible for updating the
Representation accordingly.
Agents can later query the Representation for the value of being in a state
*V(s)* or the value of taking an action in a particular state
( known as the Q-function, *Q(s,a)* ).

.. note::
    At present, it is assumed that the Linear Function approximator
    family of representations is being used.
        
.. note ::
    You may want to review the namespace / inheritance / scoping 
    `rules in Python <https://docs.python.org/2/tutorial/classes.html>`_.


Requirements 
------------

* Each Representation must be a subclass of 
  :class:`~rlpy.Representations.Representation.Representation` and call the 
  :func:`~rlpy.Representations.Representation.Representation.__init__` function 
  of the Representation superclass.

* Accordingly, each Representation must be instantiated with
  and a Domain in the ``__init__()`` function.  Note that an optional
  ``discretization`` parameter may be used by discrete Representations 
  attempting to represent a value function over a continuous space.
  It is ignored for discrete dimensions.

* Any randomization that occurs at object construction *MUST* occur in
  the :func:`~rlpy.Representations.Representation.Represenation.init_randomization`
  function, which can be called by ``__init__()``.

* Any random calls should use ``self.random_state``, not ``random()`` or 
  ``np.random()``, as this will ensure consistent seeded results during experiments.

* After your Representation is complete, you should define a unit test to ensure 
  future revisions do not alter behavior.  See rlpy/tests/test_representations 
  for some examples.


REQUIRED Instance Variables
"""""""""""""""""""""""""""

The new Representation *MUST* set the variables *BEFORE* calling the
superclass ``__init__()`` function:

#. ``self.isDynamic`` - bool: True if this Representation can add or 
   remove features during execution

#. ``self.features_num`` - int: The (initial) number of features in the representation


REQUIRED Functions
""""""""""""""""""
The new Representation *MUST* define two functions:

#. :func:`~rlpy.Representations.Representation.Representation.phi_nonTerminal`,
   (see linked documentation), which returns a vector of feature function 
   values associated with a particular state.

#. :func:`~rlpy.Representations.Representation.Representation.featureType`,
   (see linked documentation), which returns the data type of the underlying
   feature functions (eg "float" or "bool").

SPECIAL Functions
"""""""""""""""""
Representations whose feature functions may change over the course of execution
(termed **adaptive** or **dynamic** Representations) should override 
one or both functions below as needed.
Note that ``self.isDynamic`` should = ``True``.

#. :func:`~rlpy.Representations.Representation.Representation.pre_discover`

#. :func:`~rlpy.Representations.Representation.Representation.post_discover`

Additional Information
----------------------

* As always, the Representation can log messages using ``self.logger.info(<str>)``, see 
  Python ``logger`` doc. 

* You should log values assigned to custom parameters when ``__init__()`` is called.

* See :class:`~rlpy.Representations.Representation.Representation` for functions 
  provided by the superclass, especially before defining 
  helper functions which might be redundant.



Example: Creating the ``IncrementalTabular`` Representation
-----------------------------------------------------------
In this example we will recreate the simple :class:`~rlpy.Representations.IncrementalTabular.IncrementalTabular`  Representation, which 
merely creates a binary feature function f\ :sub:`d`\ () that is associated with each
discrete state ``d`` we have encountered so far.
f\ :sub:`d`\ (s) = 1 when *d=s*, 0 elsewhere, ie, the vector of feature 
functions evaluated at *s* will have all zero elements except one.
Note that this is identical to the :class:`~rlpy.Representations.Tabular.Tabular` 
Representation, except that feature functions are only created as needed, not 
instantiated for every single state at the outset.
Though simple, neither the ``Tabular`` nor ``IncrementalTabular`` representations
generalize to nearby
states in the domain, and can be intractable to use on large domains (as there
are as many feature functions as there are states in the entire space).
Continuous dimensions of ``s`` (assumed to be bounded in this Representation) 
are discretized.

#. Create a new file in your current working directory, ``IncrTabularTut.py``.
   Add the header block at the top::

        __copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
        __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
                       "William Dabney", "Jonathan P. How"]
        __license__ = "BSD 3-Clause"
        __author__ = "Ray N. Forcement"

        from rlpy.Representations.Representation import Representation
        import numpy as np
        from copy import deepcopy


#. Declare the class, create needed members variables (here an optional hash
   table to lookup feature function values previously computed), and write a 
   docstring description::

        class IncrTabularTut(Representation):
            """
            Tutorial representation: identical to IncrementalTabular

            """
            hash = None

#. Copy the __init__ declaration from ``Representation.py``, add needed parameters
   (here none), and log them.
   Assign self.features_num and self.isDynamic, then
   call the superclass constructor::

            def __init__(self, domain, discretization=20):
                self.hash           = {}
                self.features_num   = 0
                self.isDynamic      = True
                super(IncrTabularTut, self).__init__(domain, discretization)

#. Copy the ``phi_nonTerminal()`` function declaration and implement it accordingly
   to return the vector of feature function values for a given state.
   Here, lookup feature function values using self.hashState(s) provided by the 
   parent class.
   Note here that self.hash should always contain hash_id if ``pre_discover()``
   is called as required::
                
            def phi_nonTerminal(self, s):
                hash_id = self.hashState(s)
                id  = self.hash.get(hash_id)
                F_s = np.zeros(self.features_num, bool)
                if id is not None:
                    F_s[id] = 1
                return F_s

#. Copy the ``featureType()`` function declaration and implement it accordingly
   to return the datatype returned by each feature function.
   Here, feature functions are binary, so the datatype is boolean::

            def featureType(self):
                return bool

#. Override parent functions as necessary; here we require a ``pre_discover()``
   function to populate the hash table for each new encountered state::

            def pre_discover(self, s, terminal, a, sn, terminaln):
                return self._add_state(s) + self._add_state(sn)

#. Finally, define any needed helper functions::

            def _add_state(self, s):
                hash_id = self.hashState(s)
                id  = self.hash.get(hash_id)
                if id is None:
                    #New State
                    self.features_num += 1
                    #New id = feature_num - 1
                    id = self.features_num - 1
                    self.hash[hash_id] = id
                    #Add a new element to the feature weight vector
                    self.addNewWeight()
                    return 1
                return 0

            def __deepcopy__(self, memo):
                new_copy = IncrementalTabular(self.domain, self.discretization)
                new_copy.hash = deepcopy(self.hash)
                return new_copy

That's it! Now test your Representation by creating a simple settings file on the domain of your choice.
An example experiment is given below:

.. literalinclude:: ../examples/tutorial/IncrTabularTut_example.py
   :language: python
   :linenos:

What to do next?
----------------

In this Representation tutorial, we have seen how to 

* Write an adaptive Representation that inherits from the RLPy
  base ``Representation`` class
* Add the Representation to RLPy and test it


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
  representation can be added to
  the ``__init__.py`` file in the ``Representations/`` directory.
  (This allows other files to import the new representation).

* If available, please include a link or reference to the publication associated 
  with this implementation (and note differences, if any).

If you would like to add your new representation to the RLPy project, we recommend
you branch the project and create a pull request to the 
`RLPy repository <https://bitbucket.org/rlpy/rlpy>`_.

You can also email the community list ``rlpy@mit.edu`` for comments or 
questions. To subscribe `click here <http://mailman.mit.edu/mailman/listinfo/rlpy>`_.

