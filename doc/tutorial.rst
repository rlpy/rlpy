.. _tutorial:

Getting Started
===============

First Run
---------

.. tip::
    If you receive errors during any of the steps below, please refer to `install.txt` for solutions to common issues.

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

You can run the file by executing it with the python interpreter drom the rlpy
root directory::

    python examples/tutorial/gridworld.py

.. tip::
    We recommend using the IPython interpreter. Compared to the standard
    interpreter it provides color output and better help functions. It is more 
    comportable to work with in general. See `ipython homepage`_ for
    details.
    
    

    You can run the file with ipython by executing::
        
        ipython examples/tutorial/gridworld.py

    or start the interactive python shell with::
        
        ipython

    and then inside the ipython shell execute::

        %run examples/tutorial/gridworld.py

    This will not terminate the interpreter after running the file and allows
    you to inspect the objects interactively afterwards.

.. _ipython homepage: http://ipython.org

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
is terminated

The value function window shows the value function and the resulting policy.
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

Interpreting Console Outputs
----------------------------

In the console window you should see output similar to the following::
    
    647: E[0:00:01]-R[0:00:15]: Return=+0.97, Steps=33, Features = 20
    1000 >>> E[0:00:04]-R[0:00:37]: Return=+0.99, Steps=11, Features = 20
    1810: E[0:00:05]-R[0:00:23]: Return=+0.98, Steps=19, Features = 20

Each part has a specific meaning:  

.. image:: rlpy_output.png
   :width: 90 %


Note that a *performance run* or *assessment run* 
(indicated by *>>>* in the output window) tests the agent using the a greedy
policy with always choses the action with the highest value of the agent's
current value function estimate. 

.. note::
    If you see *only* performance runs as console output, 
    this simply means that your machine is completing the learning steps
    between two performance run faster than the console logging rate,
    usually 1 Hz, and does not indicate a problem.


After the cutoff of 2,000 steps specified in line 47, the experiment is 
complete, and a new window appears showing the reward earned on each 
performance run.

On this domain, an excellent policy is located almost immediately (reward 0.989).

The final plot likely shows enormous variance in reward obtained on performance runs - 
this is because the GridWorld domain has a default noise level of 0.3, meaning that 30% of 
the time, a random action will be taken; at the start, when only two actions are available, 
this corresponds to a 30% failure rate (-1 reward), even when the optimal policy has been found.

You can adjust the noise parameter "NOISE" at the top of IShouldRun.py; just be aware that the agent
does not explore (epsilon=0) during performance runs, so with 0 noise and a bad initial policy, you may 
have to sit and watch the agent execute a (likely oscillatory) deterministic policy for all 1000 steps 
of an episode!

Analyzing Data
--------------

The variable "path" in IShouldRun.py determines the directory in which all results are stored; 
IShouldRun.py defaults to ./Results/I-Should-Run.  A folder for a particular experiment is automatically 
generated in that directory as necessary based on its parameters; here, "GridWorld-Tabular-2000" for the 
GridWorld domain, Tabular representation, with 2,000 steps total. 

The data logged to the console is stored in a file "#-out.txt", where "#" is the ID of the experiment, 
here "1".  Open this "1-out.txt" and verify this. 

Data is stored in a more compact form in "#-results.txt", described below.

Now open the file "IShouldMerge.py" and run it.  
(If you get an error which includes "No directory including result was found at `Results/IShouldRun`", 
make sure you allow IShouldRun.py to run to completion, when the second figure window appears, 
approximately 30 seconds; ensure you close both figures). \n
You should now see several figures which demonstrate various performance measures of the 
experiment, such as return vs. number of steps.  The "paths" variable in this file specifies 
a directory to recursively search to generate these figures (and where to store them); if
multiple results directories are found, they are plotted simultaneously against each other
on the same figure.  If multiple results files are found in the same directory
(e.g. 1-results.txt, 2-results.txt, etc., generated using multipleRuns.py) the mean curve 
is drawn with variance around it.

You can experiment with this by re-running "IShouldRun.py", passing experiment id "2"
instead of "1" to make_experiment() on lines 78 and 83. The seed to the random number
generator is determined by this experiment id, so that results are reproducible.

A Slightly More Challenging Domain: Inverted Pendulum
-----------------------------------------------------

The small GridWorld domain is easy to understand, but does not produce meaningful
final plots because the optimal policy is learned so quickly.  We will now run an 
experiment on the Inverted Pendulum Domain, where the goal is to prevent the pendulum 
from falling below the horizontal by applying clockwise (red) or counterclockwise (black) torque. 
No reward is received for balancing and there is no control penalty; a penalty is only applied 
if the pendulum falls, after which the episode terminates.

Return to IShouldRun.py, and:

1. Near the top (line 34), change the following two variables to allow more steps for learning the domain::
    performanceChecks   = 5
    max_steps           = 8000

2. Scroll to the "Domain" section (line 58), comment the line corresponding to GridWorld and decomment
   the line corresponding to Pendulum_InvertedBalance.
Now, run IShouldRun.py and observe the new domain.  Initially, the agent fails to balance the pendulum, 
but after approximately 6000 steps, can do so reliably up to the maximum number of steps allowed for an 
episode, 300.

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


To allow the experiment to run more quickly, without visualization: 

1) Assign::

    visualize_steps     = False
    visualize_learning  = False
    visualize_performance = False

2) Assign::

    performanceChecks   = 10

3) Assign::

    max_steps           = 20000

4) Assign::
    
    domain = Pendulum_InvertedBalance(episodeCap = 3000, logger = logger)

(Step 4 extends the maximum number of steps per episode from 300 to 3000 to make the task slightly more challenging.)

Now, re-run the experiment.
Finally, return to IShouldMerge.py, comment the old "paths" variable, and decomment 
the one corresponding to the Pendulum results.  The learning up to approximately 
2,000 steps should be apparent through continuously improving performance.  
Large fluctuations in reward due to chance are smoothed by running the experiment 
many times using multipleRuns.py, described in another tutorial.

Conclusion
----------

We have seen how to run experiments, interpret visualization, and generate 
results on two classic Reinforcement Learning domains.  We have also seen
the structure of the high-level files;

IShouldRun.py <--> main.py - Identical, former has excess options removed. 
You may now try using main.py instead of IShouldRun.py, experimenting with 
different agents, representations, etc., all of which have parameters defined
at the top of main.py.
IShouldMerge.py <--> mergeRuns.py - Identical, former just has "paths" prespecified. 
You should now use mergeRuns.py to generate your results.

Finally, you should explore the rest of the structure of the framework, 
and perhaps try implementing domains and algorithms of your own.

.. epigraph::
    
    The only real mistake is the one from which we learn nothing.
    
    -- John Powell
