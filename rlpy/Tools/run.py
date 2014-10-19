import joblib
import os
import glob
import re
from rlpy.Tools import __rlpy_location__
import rlpy.Tools.condor as ct
from time import sleep
import cProfile
import pstats
import platform
import sys
import subprocess

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"

# template for executable file used to execute experiments
template = """#!/usr/bin/env python
import sys
sys.path = ["{rlpy_location}"] + sys.path
{setting_content}

{variables}
if __name__ == "__main__":
    exp = make_experiment(int(sys.argv[1]), ".", **hyper_param)
    exp.log_interval = 60
    exp.run()
    exp.save()
"""

# template for submission files of htcondor queueing system
condor_submit_template_start = """
executable = /data/scratch/cdann/anaconda/bin/python
universe = vanilla
priority = 0
Notification = Never
requirements = OpSys == "LINUX" && ARCH == "X86_64"
getenv = True"""

condor_submit_template_each_job = """
arguments = "{fn} {id}"
Error = condor/{id}.err
Log = condor/{id}.log
Output = condor/{id}.out
queue 1
"""

if sys.platform.startswith("win") or sys.platform.startswith("cygwin"):
    devnull = "nul"
else:
    devnull = "/dev/null"


def run_profiled(make_exp_fun, profile_location="Profiling",
                 out="Test.pdf", **kwargs):
    """run an experiment (without storing its results) and profiles the execution.
    A gprof file is created and a pdf with a graphical visualization of the most
    time-consuming functions in the experiment execution

    :param make_exp_fun: function that returns an Experiment instance which is then
                         executed. All remaining keyword parameters are passed to this
                         function.
    :param profile_location: directory used to store the profiling result files.
    :param out: filename of the generated pdf file.
    :param \*\*kwargs: remaining parameters passed to the experiment generator function
    """
    dat_fn = os.path.join(profile_location, "profile.dat")
    pdf_fn = os.path.join(profile_location, out)
    dot_fn = os.path.join(profile_location, "graph.txt")
    exp = make_exp_fun(**kwargs)
    cProfile.runctx('exp.run()', {}, {"exp": exp}, dat_fn)
    p = pstats.Stats(dat_fn)
    p.sort_stats('time').print_stats(5)

    gprof2dot_fn = os.path.join(__rlpy_location__, 'Tools', 'gprof2dot.py')
    if(platform.system() == 'Windows'):
        # Load the STATS and prepare the dot file for graphvis
        command = gprof2dot_fn + ' -f pstats {dat_fn} > {dot_fn}'.format(
            dat_fn=dat_fn,
            dot_fn=dot_fn)
        os.system(command)

    else:
        # Load the STATS and prepare the dot file for graphvis
        command = '/usr/bin/env python ' + gprof2dot_fn \
                  + ' -f pstats {dat_fn} > {dot_fn}'.format(
            dat_fn=dat_fn,
            dot_fn=dot_fn)
        os.system(command)

    # Call Graphvis to generate the pdf
    command = 'dot -T pdf {dot_fn} -o {pdf_fn}'.format(dot_fn=dot_fn,
                                                       pdf_fn=pdf_fn)
    os.system(command)


def get_finished_ids(path):
    """returns all experiment ids for which the result file exists in
    the given directory"""
    l = sorted([int(re.findall("([0-9]*)-results.json", p)[0])
               for p in glob.glob(os.path.join(path, "*-results.json"))])
    return l


def read_setting_content(filename):
    """reads the file content without the __main__ section
    :param filename: filename where the settings are specified"""
    setting_content = ""
    with open(filename) as f:
        lines = f.readlines()
        for l in lines:
            if "if __name__ ==" in l:
                # beware: we assume that the __main__ execution block is the
                # last one in the file
                break
            setting_content += l
    return setting_content


def prepare_directory(setting, path, **hyperparam):
    """
    Creates a directory in path with a file for executing a given
    setting. The function returns the executable python script file

    :param setting: filename which contains a make_experiment method that get
        the id and hyperparameters
        and returns an instance of Experiment ready to run
    :param path: specifies where to create the directory
    :param \*\*hyperparam: all hyperparameters passed to the setting's
        ``make_experiment()``
    :return: filename of the file to execute in path

    """
    # create file to execute
    variables = "hyper_param = dict(" + ",\n".join(["{}={}".format(k, repr(v))
                                                    for k, v in hyperparam.items()]) + ")"
    final_path = path
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    fn = os.path.join(final_path, "main.py")
    setting_content = read_setting_content(setting)
    with open(fn, "w") as f:
        f.write(template.format(setting=setting,
                                rlpy_location=__rlpy_location__,
                                variables=variables,
                                setting_content=setting_content))
    return fn


def run(filename, location, ids, parallelization="sequential",
        force_rerun=False, block=True, n_jobs=-2, verbose=10, **hyperparam):
    """
    run a file containing a RLPy experiment description (a make_experiment function)
    in batch mode. Note that the __main__ section of this file is ignored

    :param filename: file to run
    :param location: directory (does not need to exist), where all outputs and a
                     copy of the file to execute is stored
    :param ids: list of ids / seeds which should be executed
    :param parallelization: either **sequential** (running the experiment on one core
                        for each seed in sequence), **joblib** (run using multiple
                        cores in parallel, no console ouput of the individual runs)
                        or **condor** (submit jobs to a HTCondor job scheduling
                        system
    :param force_rerun: if False, seeds for which the results exists are not executed
    :param block: if True, the function returns when all jobs are done
    :param n_jobs: if parallelized with joblib, this specifies the number of cores to use
        specifying -1 means all cores, -2 means all but one cores
    :param verbose: controls the amount of outputs
    :param \**hyperaram: hyperparameter values which are passed to make_experiment
        as keyword arguments.
    """
    setting = filename
    fn = prepare_directory(setting, location, **hyperparam)

    # filter ids if necessary
    if not force_rerun:
        finished_ids = get_finished_ids(location)
        ids = [idtmp for idtmp in ids if idtmp not in finished_ids]

    if len(ids):
        # spawn jobs
        if parallelization == "joblib":
            run_joblib(fn, ids, n_jobs=n_jobs, verbose=verbose)
        elif parallelization == "condor":
            run_condor(fn, ids, force_rerun=force_rerun, block=block)
        elif parallelization == "sequential":
            run_joblib(fn, ids, n_jobs=1, verbose=verbose)


def _run_helper(fn, job_id, verbose):
    if verbose >= 15:
        out = ""
    else:
        out = "> " + devnull
    path, filen = os.path.split(fn)
    subprocess.Popen(
        "python {} {} {}".format(
            filen,
            job_id + 1,
            out),
        shell=True,
        cwd=path).wait(
    )


def run_joblib(fn, ids, n_jobs=-2, verbose=10):
    jobs = (joblib.delayed(_run_helper)(fn, i, verbose) for i in ids)
    #jobs = (joblib.delayed(os.system)(fn, i) for i in ids)
    exit_codes = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    return exit_codes


def run_condor(fn, ids,
               force_rerun=False, block=False, verbose=10, poll_duration=30):
    # create condor subdirectory
    dir = os.path.dirname(fn)
    fn = os.path.basename(fn)
    outdir = os.path.join(dir, "condor")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # exclude running jobs from ids
    if not force_rerun:
        submitted_jobs = ct.submitted_jobs_user()
        for job in submitted_jobs:
            if os.path.abspath(job["directory"]) == os.path.abspath(dir):
                if job["run_id"] in ids:
                    ids.remove(job["run_id"])
                    if verbose:
                        print "Jobs #{} already submitted".format(job["run_id"])
    if len(ids) > 0:
        # write submit file
        with open(os.path.join(outdir, "submit"), "w") as f:
            f.write(condor_submit_template_start)
            for exp_id in ids:
                f.write(condor_submit_template_each_job.format(fn=fn, exp_id=exp_id))

        exit_code = os.system(
            "cd {dir} && condor_submit condor/submit".format(dir=dir))
        if verbose:
            print "Jobs submitted with exit code", exit_code
    else:
        if verbose:
            print "All jobs have been already submitted"
    # if blocking mode in enabled, wait until all result files are there
    # WARNING: this does not recognize killed or dead jobs and would wait
    #          infinitely long
    if block:
        while(True):
            sleep(poll_duration)
            finished_ids = set(get_finished_ids(dir))
            finished_ids &= set(ids)
            if verbose > 100:
                print len(finished_ids), "of", len(ids), "jobs finished"
            if len(finished_ids) == len(ids):
                return
