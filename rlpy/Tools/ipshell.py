"""ip-shell functions"""

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"

_ip_shell = None


def ipshell():
    if _ip_shell is not None:
        return _ip_shell

    from IPython.config.loader import Config
    try:
        get_ipython
    except NameError:
        nested = 0
        cfg = Config()
        prompt_config = cfg.PromptManager
        prompt_config.in_template = 'In <\\#>: '
        prompt_config.in2_template = '   .\\D.: '
        prompt_config.out_template = 'Out<\\#>: '
    else:
        cfg = Config()
        nested = 1
    from IPython.frontend.terminal.embed import InteractiveShellEmbed

    # Now create an instance of the embeddable shell. The first argument is a
    # string with options exactly as you would type them if you were starting
    # IPython at the system command line. Any parameters you want to define for
    # configuration can thus be specified here.
    ipshell = InteractiveShellEmbed(config=cfg,
                                    banner1='Dropping into IPython',
                                    exit_msg='Leaving Interpreter, back to program.')
    _ip_shell = ipshell
    return ipshell

import signal


def interrupted(signum, frame):
    import ipdb
    ipdb.set_trace()


def ipdb_on_SIGURG():
    signal.signal(signal.SIGURG, interrupted)
