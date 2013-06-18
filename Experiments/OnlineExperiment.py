######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Experiment import *
class OnlineExperiment (Experiment):
    # Statistics are saved as : 
    # STEP            = 0 
    # RETURN          = 1 
    # CLOCK_TIME      = 2 
    # FEATURE_SIZE    = 3 
    # EPISODE_LENGTH  = 4
    # TERMINAL        = 5       # 0 = No Terminal, 1 = Normal Terminal, 2 = Critical Terminal
    max_steps           = 0     # Total number of interactions
    performanceChecks   = 0     # Number of Performance Checks uniformly scattered along the trajectory
    STATS_NUM           = 6     # Number of statistics to be saved
    LOG_INTERVAL        = 0     # Number of seconds between log prints
    def __init__(self,agent,domain,
                 id = 1,
                 max_steps = 10000, 
                 performanceChecks = 10,
                 show_all   = False, 
                 show_performance = False,
                 log_interval = 1):
        self.max_steps          = max_steps
        self.performanceChecks  = performanceChecks
        self.LOG_INTERVAL       = log_interval
        super(OnlineExperiment,self).__init__(id,agent,domain, show_all, show_performance)
    def run(self):
    # Run the online experiment and collect statistics
        self.result         = zeros((self.STATS_NUM,self.performanceChecks))
        terminal            = True
        total_steps         = 1
        eps_steps           = 1
        performance_tick    = 0
        eps_return          = 0
        start_log_time      = start_time = time()
        if self.show_all: self.domain.showLearning(self.agent.representation)
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap: 
                s           = self.domain.s0() 
                a           = self.agent.policy.pi(s)
                # Hash new state for the tabular case
                if isinstance(self.agent.representation,IncrementalTabular): self.agent.representation.addState(s)
                # Output the current status if certain amount of time has been pased
                if deltaT(start_log_time) > self.LOG_INTERVAL:
                    start_log_time  = time()
                    elapsedTime     = deltaT(start_time) 
                    print '%d: E[%s]-R[%s]: Return=%0.2f, Steps=%d, Features = %d' % (total_steps, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps), eps_return, eps_steps, self.agent.representation.features_num)
                eps_return  = 0
                eps_steps   = 0

            #Visual
            if self.show_all: self.domain.show(s,a, self.agent.representation)
            #Act,Learn,Step
            r,ns,terminal   = self.domain.step(s, a)
            na              = self.agent.policy.pi(s)
            # Hash new state for the tabular case
            if isinstance(self.agent.representation,IncrementalTabular): self.agent.representation.addState(ns)
            self.agent.learn(s,a,r,ns,na)            
            
            total_steps += 1
            eps_steps   += 1
            eps_return  += r
            s,a          = ns,na

            #Check Performance
            if  total_steps % (self.max_steps/self.performanceChecks) == 0:
                performance_return, performance_steps, performance_term = self.performanceRun(total_steps)
                elapsedTime                 = deltaT(start_time) 
                self.result[:,performance_tick] = [total_steps, 
                                                   performance_return, 
                                                   elapsedTime, 
                                                   self.agent.representation.features_num,
                                                   performance_steps,
                                               performance_term]
                print '%d >>> E[%s]-R[%s]: Return=%0.2f, Steps=%d, Features = %d' % (total_steps, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps), performance_return, performance_steps, self.agent.representation.features_num)
                start_log_time  = time()
                performance_tick += 1

        #Visual
        if self.show_all: 
            self.domain.show(s,a, self.agent.representation)
        if self.show_all or self.show_performance:
            self.result_fig.savefig('snapshot.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    def save(self,filename):
        super(OnlineExperiment,self).save(filename)
        f = open(filename,'a')
        f.write('# Column Info:\n')
        f.write('#1. Test Step\n#2. Return\n#3. Time(s)\n#4. Features\n#5. Traj-Steps\n#6. Terminal\n')
        f.close()
        #Plot Performance
        performance_fig = pl.figure(2)
        pl.plot(self.result[0,:],self.result[1,:],'-bo',lw=3,markersize=10)
        pl.xlim(0,self.result[0,-1])
        m = min(self.result[1,:])
        M = max(self.result[1,:])
        pl.ylim(m-.1*abs(M),M+.1*abs(M))
        performance_fig.savefig('performance.pdf', transparent=True, pad_inches=.1)
        
