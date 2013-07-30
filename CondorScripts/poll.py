#!/usr/bin/env python
# Functions used to poll the outputs of parallel runs on clusters
# Alborz Geramifard 2009 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?

#Inputs:
# idir : Initial Directory
# detailed: Show detailes of uncompleted files?

import os, sys, time, re 

from Script_Tools import *

def pollOne(idir, count, detailed = False, fulldetailed = False, jobinfo=None):
        if jobinfo is None:
            jobinfo = submitted_jobs_user(USERNAME, os.path.abspath(idir)) 
        if not os.path.exists(idir+'/main.py'):
            #Not a task directory
            for folder in sorted(os.listdir(idir)):
                count += 1
                if os.path.isdir(idir+'/'+folder) and folder[0] != '.':
                    pollOne(idir+'/'+folder,count,detailed,fulldetailed, jobinfo=jobinfo)
        else:                
            jobids      = set()
            #Add jobs based on CondorOutput/out/x.out
            jobs = glob.glob(idir+'/CondorOutput/log/*.log')
            for job in jobs:
                _,_,jobname = job.rpartition('/')
                jobid,_,_ = jobname.rpartition('.')
                jobids.add(eval(jobid))
            #Add jobs based on x-out.txt
            jobs = glob.glob(idir+'/*-out.txt')
            for job in jobs:
                _,_,jobname = job.rpartition('/')
                jobid,_,_ = jobname.rpartition('-')
                jobids.add(eval(jobid))
            running_jobs = filter_jobs(jobinfo, os.path.abspath(idir))
            total       = len(jobids)
            completed   = 0;
            #print jobids
            logs = []
            for jobid in jobids:
                if os.path.exists(idir+'/%d-results.txt' % jobid):                        
                    completed = completed + 1;
                else:
                    logpath = "%s/%d-out.txt" % (idir,jobid) 
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
                                log = "#%02d: %s"  % (jobid, line)
                                if fulldetailed:
                                    log = RED + log
                                logs.append(log)
                        else:
                            log = "#%02d: No output yet\n"  % (jobid)
                            logs.append(log)

            nc      = NOCOLOR
            missing = total - completed
            #print detailed, completed, total
            num_run = len([job["run_id"] for job in running_jobs if job["status"] ==2])
            num_idle = len([job["run_id"] for job in running_jobs if job["status"] ==1])
            num_held = len([job["run_id"] for job in running_jobs if job["status"] ==5])
            if len(running_jobs) or missing:
                template = "{path}: {cc}{numc}{nd}/{rc}{numr}{nd}/{ni}{numi}{nd}/{ca}{numa}{nd}"
                print template.format(path=idir.replace('./',''),
                                      cc=COMPLETED_COLOR,
                                      numc=completed,
                                      nd=nc,
                                      rc=RUNNING_COLOR,
                                      numr=num_run,
                                      numi=num_idle,
                                      ca=TOTAL_COLOR,
                                      numa=total,
                                      ni=RESUMING_COLOR)
                if num_held:
                    print "WARNING: HELD JOBS!!"

                if not fulldetailed: logs = sortLog(logs)
                for log in logs:
                    if log[-1] != '\n':
                        log = log + '\n'
                    sys.stdout.write(log)
                sys.stdout.write(nc)
            else:
                if not completed:
                    print "%s: %sEmpty.%s"  % (idir.replace('./',''), RED, nc)

if __name__ == '__main__':
    os.system('clear');
    print('*********************************************************');    
    print('************** Reporting For Duty! **********************');    
    print('*********************************************************');    
    detailed = len(sys.argv) > 1
    fulldetailed = detailed and sys.argv[1].find('+') != -1
    
    pollOne('.',0, detailed,fulldetailed)   
    
