Q&A:
==========
Q1: Which file should I use to run the framework?
A1: In the beginning, use main.py to run your experiment. You can user 
	multipleRuns.py to run the code in main.py in several threads. You can also
	import the project into Eclipse after installing the Pydev package

Q2: What does each line of output mean?
A2: "88825: E[0:01:23]-R[0:00:10]: Return=-1.00, Steps=56, Features = 174" means
	88825: 			steps of learning
	E[0:01:23]: 		Elapsed time (s)
	R[0:00:10]: 		Remaining time (s)
	Return=-1.00: 	Sum of rewards for the last episode
	Steps=56: 		Number of steps for the last episode
	Features = 174 	Number of Features used for the last episode

Q3: My code is slow, how can I improve its speed?
A3: ProfileMe.py runs the code at main.py and generates a pictorial profile of the
	resulting running time in pdf format. Each node represents proportional time
	for finishing the function, proportional time spent within the function, and
	number of times it has been called. Nodes are color coded based on their time.
	You want to spend your time boosting the running time of nodes with the highest
	proportional time spent within them shown in parentheses. As an example you can
	view Profiling/Inverted_Pendulum-TabularSarsa.pdf
	It seems phi_sa should be the place to improve the algorithm as 34.97% was spent
	within this function. 

Q4: My project does not work. Do I need to install packages?
A4: Please read install.txt

Q5: I used to plot my figures based on number of episodes why you prefer steps?
A5: The use of episode numbers does not provide accurate plots as the number of
	samples can vary within each episode. The use of steps gurantees that all
	methods saw exactly the same amount of data before being tested.
	
Q6: I have generated multipleRuns for various methods. How do I merge the results?
A6: Use mergeRuns.py. You should be able to call the mergeRuns.py with the committed
	example results. Lets assume you have the following directory structure:
				- Results/MyProject/
					- Domain-Algorithm-Representation1
					- Domain-Algorithm-Representation2
					- Domain-Algorithm-Representation3
	Set the initial path for mergeRuns to "Results/MyProject". Also use the desired
	Y and X Axes. The Y-axis can be one of the following:
		'Return': 	Sum of rewards
		'Features': 	Number of basis functions used
		'Steps': 	Length of the episode
		'Terminal': 	Did the episode finish due to reaching a terminal state
		 			or because the episode cap was reached.
	The X-axis can be:
	'Learning Steps': 	Number of interations between the agent and domain
	'Time(s)': 			Clock Time in number of seconds 
	 
	  