.. _tutorial:

Getting Started
===============

This tutorial covers the most common type of experiment in reinforcement
learning: the control experiment. An agent is supposed to find a good policy
while interacting with the domain. 

First Run
---------

Begin by looking at the file `examples/tutorial/gridworld.py`:

.. literalinclude:: ../examples/tutorial/gridworld.py
   :language: python
   :linenos:
   
The file is an example for a reinforcement learning experiment. The main
components of such an experiment is the **domain**, `GridWorld` in this case,
the **agent** (`Q_Learning`), which uses the **policy** `eGreedy` and the
value function **representation** `Tabular`. The **experiment** `Experiment` is
in charge of the execution of the experiment by handling the interaction
between the agent and the domain as well as storing the results on disk (see
also :ref:`overview`).

The function `make_experiment` gets an id, which specifies the random seeds 
and a path where the results are stored. It returns an instance of an
`Experiment` which is ready to run. In line 53, such an experiment is created
and then executed in line 54 by calling its `run` method. The three parameters
of `run` control the graphical output. The result are plotted in line 57 and
subsequently stored in line 58.

You can run the file by executing it with the ipython shell from the rlpy
root directory::

    ipython examples/tutorial/gridworld.py

.. tip::
    We recommend using the IPython shell. Compared to the standard
    interpreter it provides color output and better help functions. It is more 
    comportable to work with in general. See the `Ipython homepage`_ for
    details. 
    
.. note::
    If you want to use the standard python shell make sure the rlpy root
    directory is in the python seach path for modules. You can for example
    use::

        PYTHONPATH=. python examples/tutorial/gridworld.py

.. tip::    
    You can also use the IPython shell interactively and then run the script
    from within the shell. To do this, first start  the interactive python shell 
    with::
        
        ipython

    and then inside the ipython shell execute::

        %run examples/tutorial/gridworld.py

    This will not terminate the interpreter after running the file and allows
    you to inspect the objects interactively afterwards (you can exit the shell
    with CTRL + D)

.. _Ipython homepage: http://ipython.org

What Happens During a Control Experiment
-----------------------------------------

During an experiment, the agent does exactly in total `max_steps` learning
steps where 

    1. The agent choses an action given its (exploration) policy
    2. The domain transitions to a new state
    3. The agent observes the old and new state of the domain as well as the
       reward for this transition and improves its policy based on this new
       information

To track the performance of the agent, the quality of its current policy is
assessed `num_policy_checks` times per experiment. This is done by letting the
agent interact for `checks_per_policy` episodes with the domain in so called 
**performance runs**. During performance runs, the agent does not do any
exploration but always choses actions that it thinks is optimal. This means,
each step in a performance run consists of

    1. The agent choses an action it thinks is optimal (e.g. greedy w.r.t. its
       value function estimate
    2. The domain transitions to a new state

Note that no learning happens during performance runs. The total return for
each episode of performance runs is averaged to obtain a quality measure of the
agents policy.

Graphical Output
----------------

While running the experiment you should see two windows, one showing the domain

.. image:: gridworld_domain.png
   :width: 400px

and one showing the value function

.. image:: gridworld_valfun.png
    :width: 400px

The Domain window is a visual representation of the domain (here, *GridWorld*) 
and is useful in quickly judging or demonstrating the performance of an agent.  
In this domain, the agent (triangle) has to move
from the start (blue) to the goal (green) location in the shortest distance possible, 
while avoiding the pits (red). The agent receives -0.001 reward every step.
When it reaches the goal or a pit, it obtains rewards of +1.0 or and the episode
is terminated.

You see only the first episode of each of the ten policy assessments, since we
set `visualize_performance=1` when calling the `run` method of Experiment. You
can show more of the 100 performance runs per policy if you increase this
parameter. If you set `visualize_steps=True` also steps during learning are
shown.

The value function window shows the value function and the resulting policy. It
is shown because `visualize_learning=True`.
Notice how the policy gradually converges to the optimal, direct route which avoids pits.
After successive iterations, the agent learns the high (green) value of being in 
states that lie along the optimal path, even though they offer no immediate reward.  
It also learns the low (red) value of unimportant / undesirable states.

The set of possible actions in each grid is highlighted by arrows, where the size of arrows 
correspond to the state-action value function :math:`Q(s,a)`. 
The best action is shown as black. If the agent has not learned the optimal policy 
in some grid cells (e.g. Row 2, Column 1 in the picture above), 
it has not explored enough to learn the correct action ('left' in Row 2, Column 1).  
It likely still performs well though, since such states are only ever reached 
because of :math:`\epsilon`-greedy policy which choses random actions with
probability :math:`\epsilon=0.2`.

Most domain in RLPy have a visualization like `GridWorld` and often also a
graphical presentation of the policy or value function.

At the end of the experiment another window called *Performance* pops up and
shows a plot of the average return during each policy assessment. 

.. image:: gridworld_performance.png
   :width: 400px

As we can see the agent learns after about 500 steps to obtain on average a
reward of 0.7. The theoretically optimal reward for a single run is 0.99.
However, the noise in the domains causing the agent to move only 70% of the
times in the intended direction causes the total reward to be lower on average.
In fact, the policy learned by the agent after 500 steps is the optimal one.

Console Outputs
---------------

During execution of `examples/tutorial/gridworld.py`, you should see in the 
console window output similar to the following::
    
    647: E[0:00:01]-R[0:00:15]: Return=+0.97, Steps=33, Features = 20
    1000 >>> E[0:00:04]-R[0:00:37]: Return=+0.99, Steps=11, Features = 20
    1810: E[0:00:05]-R[0:00:23]: Return=+0.98, Steps=19, Features = 20

Each part has a specific meaning:  

.. image:: rlpy_output.png
   :width: 90 %

Lines with `>>>` are the averaged results of a policy assessment. 
Results of policy assessments are always shown. The outcome of learning
episodes is shown only every second. You might therefore see no output for
learning episodes if your computer is fast enough to do all learning steps
between two policy assessments in less than one second.


A Slightly More Challenging Domain: Inverted Pole Balancing
-----------------------------------------------------------

We will now look at how to run experiments in batch and how to analyze and
compare the performance of different methods on the same task. To this end, we
compare different value function representations on the Cart-Pole Balancing task 
with an infinite track. The task is to keep a pole balanced upright. The pole
is mounted on a cart which we can either push to the left or right.

The experimental setup is specified in `examples/tutorial/pendulum_tabular.py` with
a tabular representation and in `examples/tutorial/pendulum_rbfs.py` with radial 
basis functions (RBFs). The content of `pendulum_rbfs.py` is

.. literalinclude:: ../examples/tutorial/pendulum_rbfs.py
   :language: python
   :linenos:

Again, as the first GridWorld example, the main content of the file is a
`make_experiment` function which takes an id, a path and some more optional 
parameters and returns an :class:`Experiment.Experiment` instance. 
This is the standard format of
an RLPy experiment description and will allow us to run it in parallel on
several cores on one computer or even on a computing cluster with numerous
machines.

The content of `pendulum_tabular.py` very similar but differ in the definition
of the representation parameter of the agent. Compared to our first example,
the experiment is now executed by calling its `run_from_commandline` method.
This is a wrapper around `Experiment.run` and allows to specify the options for
visualization during the execution with command line arguments. You can for
example run::

    ipython examples/tutorial/pendulum_tabular.py -l -p

from the command line to run the experiment with visualization of the
performance runs steps, policy and value function. 

.. image:: pendulum_learning.png
   :width: 90 %

The value function (center), which plots pendulum angular rate against its angle, demonstrates 
the highly undesirable states of a steeply inclined pendulum (near the horizontal) with high 
angular velocity in the direction in which it is falling.
The policy (right) initially appears random, but converges to the shape shown, with distinct 
black (counterclockwise torque action) and red (clockwise action) regions in the first and third 
quadrants respectively, and a white stripe along the major diagonal between.  This makes intuitive
sense; if the pendulum is left of center and/or moving counterclockwise (third quadrant), for example,
a corrective clockwise torque action should certainly be applied.  The white stripe in between shows 
that no torque should be applied to a balanced pendulum with no angular velocity, or if it lies off-center 
but has angular velocity towards the balance point.

If you pass no command line
arguments, no visualization is shown and only the performance graph at the end 
is produced. For an explination of each command line argument type::
 
    ipython examples/tutorial/pendulum_tabular.py -h

When we run the experiment with the tabular representation, we see that the
pendulum can be balanced sometimes, but not reliably.

In order to properly assess the quality of the learning algorithm using this
representation, we need to average over several independent learning sequences.
This means we need to execute the experiment with different seeds.

Running Experiments in Batch
----------------------------

The module :mod:`Tools.run` provides several functions that are helpful for
running experiments. The most important one is :func:`Tools.run.run`.

It allows us to run a specific experimental setup specified by a
`make_experiment` function in a file with multiple seeds in parallel. For
details see :func:`Tools.run.run`.

You find in `examples/tutorials/run_pendulum_batch.py` a short script with the
following content:

.. literalinclude:: ../examples/tutorial/run_pendulum_batch.py
   :language: python
   :linenos:

This script first runs the inverted pendulum experiment with radial basis
functions ten times with seeds 1 to 10. Subsequently the same is done for the
experiment with tabular representation. Since we specified 
`parallelization=joblib`, the joblib library is used to run the experiment in
parallel on all but one core of your computer.
The execution of this script with::

    ipython examples/tutorial/run_pendulum_batch.py

might take a few minutes depending on your hardware. 

Analyzing Results
-----------------

Running experiments via :func:`Tools.run.run` automatically saves the results 
to the specified path. If we run an :class:`Experiments.Experiment` instance
directly, we can store the results on disc with the
:func:`Experiments.Experiment.save` method. The outcomes are then stored in
the directory that is passed during initialization. The filename has the format
`XXX-results.json` where `XXX` is the id / seed of the experiment. The results
are stored in the JSON format that look for example like::

    {"learning_steps": [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], 
     "terminated": [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.3, 0.3, 0.0, 0.7, 0.0], 
     "return": [-1.0, -1.0, -1.0, -1.0, -0.9, -0.8, -0.3, -0.3, 0.0, -0.7, 0.0], 
     "learning_time": [0, 0.31999999999999995, 0.6799999999999998, 1.0099999999999998, 1.5599999999999996, 2.0300000000000002, 2.5300000000000002, 2.95, 3.3699999999999983, 3.7399999999999993, 4.11], 
     "num_features": [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400], 
     "learning_episode": [0, 45, 71, 85, 99, 104, 110, 121, 136, 144, 152], 
     "discounted_return": [-0.6646429809896579, -0.529605466143065, -0.09102296558580342, -0.2085618862726307, -0.012117452394591856, -0.02237266958836346, -0.012851215851463843, -0.0026252190655709274, 0.0, -0.0647935684347749, 0.0], 
     "seed": 1, 
     "steps": [9.0, 14.1, 116.2, 49.3, 355.5, 524.2, 807.1, 822.4, 1000.0, 481.0, 1000.0]}

The measurements of each assessment of the learned policy is stored
sequentially under the corresponding name.
The module :mod:`Tools.results` provides a library of functions and classes that 
simplify the analysis and visualization of results. See the the api documentation
for details.

To see the different effect of RBFs and tabular representation on the
performance of the algorithm, we will plot their average return for each policy
assessment. The script saved in `examples/tutorial/plot_result.py` shows us
how:

.. literalinclude:: ../examples/tutorial/plot_result.py
   :language: python
   :linenos:

First, we specify the results we specify the directories where the results are
stored and give them a label, here *RBFs* and *Tabular*. Then we create an
instance of :class:`Tools.results.MultiExperimentResults` which loads all
corresponding results an let us analyze and transform them. In line 7, we plot
the average return of each method over the number learning steps done so far.
Finally, the plot is saved in `./Results/Tutorial/plot.pdf` in the lossless pdf
format. When we run the script, we get the following plot

.. image:: pendulum_plot.png
   :width: 500px

The shaded areas in the plot indicate the standard error of the sampling mean.
We see that with radial basis functions the agent is able to perform perfectly
after 2000 learning steps, but with the tabular representation, it stays at a
level of -0.4 return per episode. Since the value function only matters around
the center (zero angle, zero velocity), radial basis functions can capture the
necessary form there much more easily and therefore speed up the learning
process.

.. warning:: The rest of this tutorial is outdated!

Tuning Hyper-Parameters
-----------------------

The behavior of each component of an agent can be drastically modified by its
parameters (or hyper-parameters, in contrast to the parameters of the value
function that are learned). The module





.. epigraph::
    
    The only real mistake is the one from which we learn nothing.
    
    -- John Powell
