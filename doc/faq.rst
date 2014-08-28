.. _faq:

Frequently Asked Questions (FAQ)
================================

How do I use the framework?
---------------------------

You can have a look at :ref:`tutorial` or the `examples` directory where you
find many ready-to-run examples of reinforcement learning experiments.


What does each line of output mean?
-----------------------------------

See documentation in the Getting Started section of the :ref:`tutorial`.

    88825: E[0:01:23]-R[0:00:10]: Return=-1.00, Steps=56, Features = 174

+-----------------+----------------------------------------------+
| Field           |  Meaning                                     |
+=================+==============================================+
| 88825           | steps of learning                            |
+-----------------+----------------------------------------------+
| E[0:01:23]      | Elapsed time (s)                             |
+-----------------+----------------------------------------------+
| R[0:00:10]      | Remaining time (s)                           |
+-----------------+----------------------------------------------+
| Return=-1.00    | Sum of rewards for the last episode          |
+-----------------+----------------------------------------------+
| Steps=56        | Number of steps for the last episode         |
+-----------------+----------------------------------------------+
| Features = 174  | Number of Features used for the last episode |
+-----------------+----------------------------------------------+

My code is slow, how can I improve its speed?
---------------------------------------------

You can use the :func:`rlpy.Tools.run.run_profiled` function which takes a
`make_experiment` function and generates a pictorial profile of the
resulting running time in pdf format (see api doc for details on where to
find this files). 
Each node represents proportional time
for finishing the function, proportional time spent within the function, and
number of times it has been called. Nodes are color coded based on their time.
You want to spend your time boosting the running time of nodes with the highest
proportional time spent within them shown in parentheses. As an example you can
look at ``Profiling/Example.pdf``

My project does not work. Do I need to install packages?
--------------------------------------------------------

Please see the :ref:`Install page <install>`.

I used to plot my figures based on number of episodes, why do you prefer steps?
-------------------------------------------------------------------------------
The use of episode numbers does not provide accurate plots as the number of
samples can vary within each episode. The use of steps gurantees that all
methods saw exactly the same amount of data before being tested.
