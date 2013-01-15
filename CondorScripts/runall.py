#!/usr/bin/python
# Functions used to remove all unfinished jobs and rerun them.
# The prerequisit is to not have any jobs running on the server
# Alborz Geramifard 2009 MIT
# Modified Robert H Klein 2013 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?

#Inputs:
# exp_num: Number of experiment runs that are required to be finished
#+force: Just run the god damn thing! => Ignore all warnings. (default = false)

import os, sys, time, re, string
#Add all paths
path = '.'
while not os.path.exists(path+'/Tools.py'):
    path = path + '/..'
sys.path.insert(0, os.path.abspath(path))

RL_PYTHON_ROOT = path 
from Script_Tools import * 

TEST = False # This value is used to avoid actually doing anything, so we can check the program
USERNAME='agf'
FINALFILE='result'


def submit(id):
    #Submit one task to condor using id
    if n > 0:
        condrun='mkdir -p CondorOutput;' + \
                'cd CondorOutput;' + \
                'mkdir -p log;' +\
                'mkdir -p err;' +\
                'mkdir -p out;' +\
                'cd ..;' +\
                'condor_submit'+\
                 ' -a arguments = main('+str(id)+')' + RL_PYTHON_ROOT+'/CondorScripts/submit_script.sh' +\
                 ' -a \'Error = CondorOutput/err/'+str(id)+'.err\''+\
                 ' -a \'Log = CondorOutput/log/'+str(id)+'.log\''+\
                 ' -a \'Output = CondorOutput/out/'+str(id)+'.out\''+\
                 RL_PYTHON_ROOT+'/CondorScripts/submit_script.sh'
        sysCall(condrun)
     
def searchNSubmit(idir,exp_num,answered,respawnjobs):
        print idir
        #See if this directory is a potential experiment 
        if not os.path.exists(idir+'/main.py') or os.path.exists(idir+'/Domains'):
            #print ' (!) ' + idir + '  not an experiment.'
            for folder in os.listdir(idir):
                newdir = idir+'/'+folder
                if os.path.isdir(newdir):
                    [answered,respawnjobs] = searchNSubmit(newdir,exp_num,answered,respawnjobs)
            return [answered,respawnjobs]
        
#        if PURGEJOBS:
#        	sysCall("rm -rf " +idir+'/CondorOutput')

        if idir != '.':
            print "========================="
            print 'Experiment: '+ idir
                            
        total           = 0
        completed       = 0
        running         = 0
        respawned       = 0
        
        #Going inside directory
        currentdir = os.getcwd()
        os.chdir(idir) 

        
        allouts             = glob.glob('*-out.txt')
        ran_num             = len(allouts)
        completed_results   = glob.glob('*-results.txt')
        completed           = len(completed_results)
        for out in allouts: 
            
            #Break if we have enough runs
            if completed+respawned+running >= exp_num:
            	   # We have extra jobs than exp_num => Remove those without results
		       continue
            
            id,_,_ = job.rsplit('-')
            if not os.path.exists('%d-results.txt'%id):                        
                while not answered:
                    answer=raw_input('(!) Respawn Jobs ? => (Y/N)')
                    if answer.lower() == 'y':
                        respawnjobs  = True
                        answered    = True

                    if answer.lower() == 'n':
                        respawnjobs  = False
                        answered    = True
                            
                if respawnjobs:
                    #Show content
                        command           = "tail -n 1 " + job
                        sysCommandHandle  = os.popen(command)
                        for line in sysCommandHandle:
                            print "Job #" + id + ": " + line,
                        submit(eval(id))
                        print RESUMING_COLOR+">>> Resumed Job #"+id+NOCOLOR
                        respawned = respawned + 1
                else:
                    running = running + 1
                        
        	
        # Submit extra jobs as needed
        # Here there is no more jobs to respawn so we have to create new ones
        extraNeed       = exp_num-completed-respawned-running
        newSubmission   = 0
        jobid           = 1
        while newSubmission < extraNeed:
            if os.path.exists('%d-out.txt',jobid):
                jobid = jobid + 1
                continue
            
            newSubmission = newSubmission + 1
            submit(jobid)
            print YELLOW+">>> Submitted Job #"+str(jobid)+NOCOLOR 
                
        print "---------------------"
        print "Completed:\t%d" % completed
        print "Running:\t%d" % (running)
        print "Resuming:\t%d" % (respawnd)
        print "New Submission:\t%d" % (newSubmission)

        #Return to the directory we started at
        os.chdir(currentdir)
        return [answered, respawnjobs]
    
def sysCall(cmd):
    if TEST:
        print cmd
    else:
        sys.os(cmd)

def rerun(idir,exp_num):
        
        # Check queue by running condor_q and extracting jobs
        command     = 'condor_q ' + USERNAME
        result      = os.popen(command)
        P           = re.compile('[0-9]+ jobs')

        runningJobs = 100
        for l in result:
            Matchlist   = P.findall(l)
            if len(Matchlist) == 1:
                jobstr      = Matchlist[0]
                end_index   = jobstr.find(' ')
                jobstr      = jobstr[:end_index]
                runningJobs = eval(jobstr)
        
        if runningJobs > 0:
            print ">>> Found running jobs ("+str(runningJobs)+") For user " + USERNAME + "."
        
            gotanswer1 = False or force
            while not gotanswer1:
                answer=raw_input('(!) (K)ill / (I)gnore / (Q)uit ?')
                
                if answer.lower() == 'q':
                    sys.exit(0);
                
                if answer.lower() == 'k':
                    print "Killing all jobs..." 
                    command   = 'condor_rm ' + USERNAME
                    result    = sysCall(command)
                    gotanswer1 = True;
                    
                if answer.lower() == 'i':
                    print "Ignoring all running jobs..."
                    gotanswer1 = True;
        else:
            respawnjobs  = True #If no task is running try to respawn jobs
            print ">>> No job found for user " + USERNAME + "."
        
        #Start Searching and Purging
        searchNSubmit(idir,exp_num,False,False)
    
if __name__ == '__main__':
    
    os.system('clear');

    force = False
    runs = 30
    
    if len(sys.argv) > 1:
    	runs = eval(sys.argv[1])
    
    if len(sys.argv) > 2:
        force = True

    print('*****************************************************');    
    print('******************* Run %d experiments **************' % (runs));    
    print('*****************************************************');     
    if force: print ('>>> FORCED MODE: Ignoring all warnings! <<<')
    
    rerun('.',runs)   
    
