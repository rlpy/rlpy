# The following returns the directory in which this file is located; thus
# even if the RL-Python directory is moved, the environment variable is
# updated every time a shell is opened.
export RL_PYTHON_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
