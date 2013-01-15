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

from Tools import * 

TEST = False # This value is used to avoid actually doing anything, so we can check the program
USERNAME='rhklein'
FINALFILE='result'


def submit(n,jobdir):
    #Submit one task to condor
    if n > 0:
        condrun='mkdir -p CondorOutput;' + \
                'cd CondorOutput;' + \
                'mkdir -p log;' +\
                'mkdir -p err;' +\
                'mkdir -p out;' +\
                'cd ..;' +\
                'condor_submit'+\
                 ' -a \"arguments = -nodisplay -nosplash -r main('+str(n)+') -logfile '+jobdir+'/log.txt' +'\" ~/myscripts/MIT/runpy-new.sh' +\
                 ' -a \'Error = CondorOutput/err/'+str(n)+'.err\''+\
                 ' -a \'Log = CondorOutput/log/'+str(n)+'.log\''+\
                 ' -a \'Output = CondorOutput/out/'+str(n)+'.out\''+\
                 ' ~/myscripts/MIT/runpy-new.sh'
                # 128.30.65.35 Error: The input character is not valid in MATLAB statements or expressions. Referring to (') 
                # ' -a \'arguments= -nodisplay -nosplash -r \\\"main('+str(n)+')\\\" -logfile '+jobdir+'/log.txt \''+\
                # 128.30.29.36  Error: syntax error near unexpected token `(' 
                #'-a \"arguments = -nodisplay -nosplash -r main('+str(n)+') -logfile '+jobdir+'/log.txt' +'\" ~/myscripts/MIT/runpy-new.sh' +\
                # Condor Server: Found illegal unescaped double-quote: "main(1)" -logfile Results/Parallel/1/log.txt
                #'-a \'arguments= -nodisplay -nosplash -r "main('+str(n)+')" -logfile '+jobdir+'/log.txt \''+\
                # 128.30.112.26 Error simply outputs: main(1) as a string!
                #'-a \"arguments= -nodisplay -nosplash -r \'main('+str(n)+')\' -logfile '+jobdir+'/log.txt\" '+\
                # Error: Found illegal unescaped double-quote: "main(1)" -logfile Results/Parallel/1/log.txt
                #'-a \'arguments= -nodisplay -nosplash -r \"main('+str(n)+')\" -logfile '+jobdir+'/log.txt\' '+\

#                condor_submit -a \"queue '+str(n)+'\" ~/myscripts/MIT/runpy.sh' 
        # 
        if not TEST:
            os.popen(condrun)
        else:
        	print condrun
     
def searchNSubmit(idir,exp_num,answered,resumejobs):
        
            
        #See if this directory is a potential experiment 
        if not os.path.exists(idir+'/main.py') or os.path.exists(idir+'/Domains'):
            #print ' (!) ' + idir + '  not an experiment.'
            for folder in os.listdir(idir):
                newdir = idir+'/'+folder
                if os.path.isdir(newdir):
                    [answered,resumejobs] = searchNSubmit(newdir,exp_num,answered,resumejobs)
            return [answered,resumejobs]
        
#        if PURGEJOBS:
#        	sysCall("rm -rf " +idir+'/CondorOutput')

        if idir != '.':
            print "========================="
            print 'Experiment: '+ idir
                            
        total           = 0
        completed       = 0
        running         = 0
        resumed         = 0
        
        #Going inside directory
        currentdir = os.getcwd()
        os.chdir(idir) 

        if os.path.exists(PARALLEL_SUBDIR):
            
            jobs    = os.listdir(PARALLEL_SUBDIR)
            total   = len(jobs)
            for job in jobs: 
                
                jobdir = PARALLEL_SUBDIR+job

                #Break if we have enough runs
                if completed+resumed+running >= exp_num:
                	# We have extra jobs than exp_num => Remove those without results
		            if not os.path.exists(jobdir +'/' + FINALFILE) and not os.path.exists(jobdir +'/Backup/Backup.mat'):                        
            			sysCall("rm -r " + jobdir)
            			print RED+">>> Removed unfinished extra Job #"+job+NOCOLOR
        			continue
                
                existing = os.listdir(jobdir)
                if os.path.exists(jobdir +'/' + FINALFILE):                        
                    completed = completed + 1;
                    if os.path.exists(jobdir+'/Backup'):
                        print RED+">>> Purged Backup for Job # "+job+NOCOLOR
                        sysCall("rm -r " + jobdir+'/Backup')
                else:
                    while not answered:
                        answer=raw_input('(!) Resume Directories ? => (Y/N)')
                        if answer.lower() == 'y':
                            resumejobs  = True
                            answered    = True

                        if answer.lower() == 'n':
                            resumejobs  = False
                            answered    = True
                                
                    if resumejobs:
                        #Show content
                        if os.path.exists(jobdir+'/log.txt'):
                            command           = "tail -n 1 " + jobdir + "/log.txt"
                            sysCommandHandle  = os.popen(command)
                            for line in sysCommandHandle:
                                print "Job #" + job + ": " + line,
                        submit(eval(job),jobdir)
                        print RESUMING_COLOR+">>> Resumed Job #"+job+NOCOLOR
                        resumed = resumed + 1
                    else:
                        running = running + 1
                        
        else:
        	os.makedirs(PARALLEL_SUBDIR)
        	
        # Submit extra jobs as needed
        # Here there is no more jobs to resume so we have to create new ones
        extraNeed       = exp_num-completed-resumed-running
        newSubmission   = 0
        job             = 1
        while newSubmission < extraNeed:
            jobdir = PARALLEL_SUBDIR + str(job)
            if os.path.exists(jobdir):
                job = job + 1
                continue
            
            newSubmission = newSubmission + 1
            if not TEST:
                os.mkdir(jobdir)
            submit(job,jobdir)
            print YELLOW+">>> Submitted Job #"+str(job)+NOCOLOR 
                
        print "---------------------"
        print "Completed:\t%d" % completed
        print "Running:\t%d" % (running)
        print "Resuming:\t%d" % (resumed)
        print "New Submission:\t%d" % (newSubmission)

        #Return to the directory we started at
        os.chdir(currentdir)
        return [answered, resumejobs]
    
def sysCall(cmd):
    if TEST:
        print cmd
    else:
        sys.os(cmd)

def rerun(idir,exp_num):
        
        # Check queue by running condor_q and extracting jobs
        command     = 'condor_q ' + USERNAME
        if not TEST:    result = os.popen(command)
        else:
            print 'os.popen('+command+')'
            result = arange(100)
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
            resumejobs  = True #If no task is running try to resume jobs
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
    
