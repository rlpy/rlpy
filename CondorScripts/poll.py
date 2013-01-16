#!/usr/bin/python
# Functions used to poll the outputs of parallel runs on clusters
# Alborz Geramifard 2009 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?

#Inputs:
# idir : Initial Directory
# detailed: Show detailes of uncompleted files?

import os, sys, time, re 

from Script_Tools import *

def pollOne(idir, count, detailed = False, fulldetailed = False):
        if not os.path.exists(idir+'/main.py'):
            #Not a task directory
            for folder in os.listdir(idir):
                count += 1
                if os.path.isdir(idir+'/'+folder) and folder[0] != '.':
                    pollOne(idir+'/'+folder,count,detailed,fulldetailed)
        else:                
            jobs        = glob.glob(idir+'/*-out.txt')
            total       = len(jobs)
            completed   = 0;
        
            logs = []
            for job in jobs:
                _,_,jobname = job.rpartition('/')
                jobid,_,_ = jobname.rpartition('-')
                if os.path.exists(idir+'/%s-results.txt' % jobid):                        
                    completed = completed + 1;
                else:
                    logpath = "%s/Log/%s.txt" % (idir,jobid) 
                    if detailed and os.path.exists(logpath):
                        if fulldetailed:
                            command = "tail -n 30 " + logpath
                        else:
                            command = "tail -n 1 " + logpath
                        
                        sysCommandHandle = os.popen(command)

                        gotSomething = False
                        lines = []
                        for line in sysCommandHandle:
                            lines.append(line)
                            if len(line) > 1:
                                gotSomething = True
                            
                        if gotSomething:
                            for line in lines:
                                log = "#%02d: %s"  % (eval(jobid), line)
                                if fulldetailed:
                                    log = RED + log
                                logs.append(log)
                        else:
                            log = "#%02d: No output yet\n"  % (eval(job))
                            logs.append(log)

            nc      = NOCOLOR
            running = total - completed
            #print detailed, completed, total
            if running:
                print"(%s%d%s/%s%d%s) %s"  % (RUNNING_COLOR,completed,nc,TOTAL_COLOR,total,nc,idir.replace('./',''))

                if not fulldetailed: logs = sortLog(logs)
                for log in logs:
                    if log[-1] != '\n':
                        log = log + '\n'
                    sys.stdout.write(log)
                sys.stdout.write(nc)
            else:
                print "%s Done: (%d/%d) %s%s"  % (COMPLETED_COLOR,completed,total,idir.replace('./',''), nc)

if __name__ == '__main__':
    os.system('clear');
    print('*********************************************************');    
    print('************** Reporting For Duty! **********************');    
    print('*********************************************************');    
    detailed = len(sys.argv) > 1
    fulldetailed = detailed and sys.argv[1].find('+') != -1
    pollOne('.',0, detailed,fulldetailed)   
    
