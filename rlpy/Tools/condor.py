"""Useful functions for interfacing with condor"""

import subprocess
import re

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


def submitted_jobs_user(username=None, path=None):
    """
    returns info on submitted jobs for given user
    as a list of dictionaries
    """
    if username is None:
        username = get_current_user()
    output = subprocess.Popen(['condor_q', username, "-long"],
                              stdout=subprocess.PIPE).communicate()[0]
    jobs = output.strip().split("\n\n")
    joblist = []
    for job in jobs:
        j = re.findall("ClusterId = (.*)\n", job)
        if len(j) == 0 and len(jobs) <= 1:
            return []
        job_id = j[0]
        directory = re.findall('Iwd = "(.*)"\n', job)[0]
        run_id = int(re.findall('UserLog = ".*/([0-9]*).log"\n', job)[0])
        status = int(re.findall('\nJobStatus = ([0-9]*)\n', job)[0])
        if path is not None and not directory.startswith(path):
            continue
        joblist.append(
            dict(job_exp_id=job_id,
                 directory=directory,
                 run_exp_id=run_id,
                 status=status))
    return joblist


def get_current_user():
    """
    returns the username of the user that executes this file
    """
    return (
        subprocess.Popen(
            ['whoami'],
            stdout=subprocess.PIPE).communicate()[0].strip()
    )
