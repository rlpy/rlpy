Getting Started
===============

First Run
---------

.. tip::
    If you receive errors during any of the steps below, please refer to `install.txt` for solutions to common issues.

Begin by looking at the file `cases/tutorial/gridworld.py`:

.. literalinclude:: ../cases/tutorial.py
   :language: python
   :linenos:
   
The most important part is the `make_experiment` function which every file specifying an
experimental setup in RLPy should contain.

You will notice a series of parameters at the top of the file, 
followed by assignments to each of the functional components 
shown in :ref:`The Big Picture <big_picture>`: domain, representation, policy, agent, and experiment.
Leave these alone for now; just run the file as-is.  You should see something like the following:

.. image:: gridWorld_learning.png
   :width: 90 %

This is a visual representation of the domain (here, *GridWorld*) and is useful in quickly judging or
demonstrating the performance of an experiment.  The objective on this domain is to move the agent (triangle) 
from the start (blue) to the goal (green) location in the shortest distance possible, while avoiding the 
pits (red); -0.001 reward is applied for every step, and reaching the goal or pit regions give 
rewards of +1.0 and -1.0 respectively, terminating the episode.

On the left, you can see the learned policy in action after each 200 steps of training data.  
Notice how the policy gradually converges to the optimal, direct route which avoids pits.
On the right, you can see the representation of the value function overlayed on the domain.  
Notice how after successive iterations, the agent learns the high (green) value of being in 
states that lie along the optimal path, even though they offer no immediate reward.  
It also learns the low (red) value of unimportant / undesirable states.

The set of possible actions in each grid is highlighted by arrows, where the size of arrows 
correspond to the :math:`Q(s,a)`. The best action is shown as black. 
If the agent hasn't learned the optimal policy in some grid cells (e.g. Row 2, Column 1), 
it has not explored enough to learn the correct action ('left' in Row 2, Column 1).  
It likely still performs well though, since such states are only ever reached because of random noise in the agent's actions.

Each domain in RLPy offers visualization like that shown on the left, and where possible, 
a representation of the value function like that shown on the right as well.

Interpreting Output
-------------------

In the console window you should see output similar to the following::
    
    647: E[0:00:01]-R[0:00:15]: Return=+0.97, Steps=33, Features = 20
    1000 >>> E[0:00:04]-R[0:00:37]: Return=+0.99, Steps=11, Features = 20
    1810: E[0:00:05]-R[0:00:23]: Return=+0.98, Steps=19, Features = 20

Each part has a specific meaning:  

.. image:: rlpy_output.png
   :width: 90 %


Note that a *performance run* (indicated by *>>>* in the output window) tests 
the agent using its latest policy, without any exploration or modifications that 
might be used during learning (such as the randomization of the episilon-greedy policy). 
The visualization shows these performance runs.

.. note::
    If you see *only* performance runs as console output, this simply means that your machine 
    is completing the learning before the performance run faster than the console logging rate,
    usually 1 Hz, and does not indicate a problem.


After the cutoff of 2,000 steps specified in `cases/tutorial/gridworld.py`, the experiment is 
complete, and a second figure window appears, showing the reward earned on each performance run.  
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
