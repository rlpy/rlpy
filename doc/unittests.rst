.. _unittests:

Creating a Unit Test
=====================

This tutorial briefly describes how to create a unit test for your new module.

.. Below taken directly from Domain.py

Python Nose
===========

RLPy uses `nose <https://nose.readthedocs.org/en/latest/>`_ to perform unit 
tests.
The syntax is::

    nosetests <directory or file to test>

If a directory is supplied, ``nose`` attempts to recursively 
locate all files that it thinks contain tests.
Include the word **test** in your filename and ``nose`` will search the file
for any methods that look like tests; ie, again, include **test** in your method
name and nose will execute it.
Note for example, the tabular domain can be tested by running::

    nosetests rlpy/tests/test_representations/test_Tabular.py

And that all representations (that have tests defined)
can be tested by running::

    nosetests rlpy/tests/test_representations/

And that in fact all modules with tests
can be tested by running::

    nosetests rlpy/tests/

.. warning::
    The last command may take several minutes to run.




Unit Test Guidelines
====================
There are no technical requirements for the unit tests; they should ensure
correct behavior, which is unique to each class.

In general, each method should start with *test* and contain a series of 
``assert`` statements that verify correct behavior.

Test:
    * preconditions
    * postconditions
    * consistency when using a seeded random number generator
    * etc.

Pay special attention to corner cases, such as 
when a Representation has not yet received any data or when an episode is reset.


Example: Tabular
================
Open ``rlpy/tests/test_representations/test_Tabular.py``
Observe the filename includes the word *test*, as does each method name.

Many, many tests are possible, but the author identified the most pertinent 
ones as::

    * Ensure appropriate number of cells are created
    * Ensure the correct binary feature is activated for a particular state
    * Ensure correct discretization in continuous state spaces

These have each been given a test as shown in the file.

The integrity of the module is always tested with a statement of the form
``assert <module quantity> == <expected/known quantity>``.

For
example, the code::

    mapname=os.path.join(mapDir, "4x5.txt") # expect 4*5 = 20 states
    domain = GridWorld(mapname=mapname)
    rep = Tabular(domain, discretization=100)
    assert rep.features_num == 20
    rep = Tabular(domain, discretization=5)
    assert rep.features_num == 20

creates a 4x5 GridWorld and ensures that Tabular creates the correct number 
of features (20), and additionally tests that this holds true even when bogus
``discretization`` parameters are passed in. 


What to do next?
----------------

In this unit testing tutorial, we have seen how to 

* Write a successful unit test
* Use ``nosetests`` to run the tests

You should write unit tests for any new modules you create.
Feel free to modify / extend existing unit tests as well - there is always more
to test!

