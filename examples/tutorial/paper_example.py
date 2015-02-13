import rlpy
#### Domain ####
domain = rlpy.Domains.InfCartPoleBalance()
### Agent ####
representation 	= rlpy.Representations.Tabular(domain, discretization=20)
policy = rlpy.Policies.eGreedy(representation, epsilon=0.1)
agent = rlpy.Agents.SARSA(policy, representation, domain.discount_factor)
### Experiment ####
experiment = rlpy.Experiments.Experiment(agent, domain, max_steps=100)
experiment.run()
experiment.save()
