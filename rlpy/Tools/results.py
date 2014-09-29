"""Parsing, extracting statistics and plotting of experimental results."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
import os
import numpy as np
import glob

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


def _thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.0fk' % (x * 1e-3)

thousands_formatter = FuncFormatter(_thousands)

#: default labels for result quantities
default_labels = {"learning_steps": "Learning Steps",
                  "return": "Average Return",
                  "discounted_return": "Discounted Return",
                  "learning_time": "Computation Time"}
#: default colors used for plotting
default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
#: default markers used for plotting
default_markers = [
    'o',
    'v',
    '8',
    's',
    'p',
    '*',
    '<',
    'h',
    '^',
    'H',
    'D',
    '>',
    'd']


def thousand_format_xaxis():
    """set the xaxis labels to have a ...k format"""
    plt.gca().xaxis.set_major_formatter(thousands_formatter)


def load_single(filename):
    """
    loads and returns a single experiment stored in filename
    returns None if file does not exist
    """
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        result = json.load(f)
    return result


def get_all_result_paths(path, min_num=1):
    """
    scan all subdirectories of a list of paths if they contain
    at least min_num results
    the list of paths with results are returned
    """
    exp_paths = []
    for p in os.walk(path):
        dirname = p[0]
        if contains_results(dirname, min_num):
            exp_paths.append(dirname)


def load_results(path):
    """
    returns a dictionary with the results of each run of an experiment stored
    in path
    The keys are the seeds of the single runs
    """
    results = {}
    for fn in glob.glob(os.path.join(path, '*-results.json')):
        cur_result = load_single(fn)
        results[cur_result["seed"]] = cur_result
    return results


def contains_results(path, min_num=1):
    """
    determines whether a directory contains at least min_num results or not
    """
    return len(glob.glob(os.path.join(path, '*-results.json'))) >= min_num


def avg_quantity(results, quantity, pad=False):
    """
    returns the average and standard deviation and number of observations
    over all runs of a certain quantity.
    If pad is true, missing entries for runs with less entries are filled with the last value
    """
    length = max([len(v[quantity]) for v in results.itervalues()])
    mean = np.zeros(length)
    std = np.zeros(length)
    num = np.zeros(length, dtype="int")
    last_values = {}
    for i in xrange(length):
        for k, v in results.iteritems():
            if len(v[quantity]) > i:
                last_values[k] = v[quantity][i]
                num[i] += 1
            else:
                if pad:
                    num[i] += 1
                else:
                    last_values[k] = 0.
            mean[i] += last_values[k]
        if num[i] > 0:
            mean[i] /= num[i]

        for k, v in results.iteritems():
            if len(v[quantity]) > i:
                last_values[k] = v[quantity][i]
                num[i] += 1
            else:
                if pad:
                    num[i] += 1
                else:
                    last_values[k] = 0.
            std[i] += (last_values[k] - mean[i]) ** 2
        if num[i] > 0:

            std[i] /= num[i]
        std[i] = np.sqrt(std[i])
    return mean, std, num


def first_close_to_final(x, y, min_rel_proximity=0.05):
    """
    returns the chronologically first value of x where
    y was close to min_rel_proximity (y[-1] - y[0]) of
    the final value of y, i.e., y[-1].
    """
    min_abs_proximity = (y[-1] - y[0]) * min_rel_proximity
    final_y = y[-1]
    for i in xrange(len(x)):
        if abs(y[i] - final_y) < min_abs_proximity:
            return x[i]


def add_first_close_entries(results, new_label="95_time",
                            x="time", y="return", min_rel_proximity=0.05):
    """
    adds an entry to each result for the time required to get within
    5% of the final quantity.
    returns nothing as the results are added in place
    """
    for v in results.itervalues():
        v[new_label] = first_close_to_final(x, y, min_rel_proximity)


class MultiExperimentResults(object):

    """provides tools to analyze, compare, load and plot results of several
    different experiments each stored in a separate path"""

    def __init__(self, paths):
        """
        loads the data in paths
        paths is a dictionary which maps labels to directories
        alternatively, paths is a list, then the path itself is considered
        as the label
        """
        self.data = {}
        if isinstance(paths, list):
            paths = dict(zip(paths, paths))
        for label, path in paths.iteritems():
            self.data[label] = load_results(path)

    def plot_avg_sem(
            self, x, y, pad_x=False, pad_y=False, xbars=False, ybars=True,
            colors=None, markers=None, xerror_every=1,
            legend=True, **kwargs):
        """
        plots quantity y over x (means and standard error of the mean).
        The quantities are specified by their id strings,
        i.e. "return" or "learning steps"

        ``pad_x, pad_y``: if not enough observations are present for some results,
        should they be filled with the value of the last available obervation?\n
        ``xbars, ybars``: show standard error of the mean for the respective 
        quantity colors: dictionary which maps experiment keys to colors.\n
       ``markers``: dictionary which maps experiment keys to markers.
        ``xerror_exery``: show horizontal error bars only every .. observation.\n
        ``legend``: (Boolean) show legend below plot.\n

        Returns the figure handle of the created plot
        """
        style = {
            "linewidth": 2, "alpha": .7, "linestyle": "-", "markersize": 7,
        }
        if colors is None:
            colors = dict([(l, default_colors[i % len(default_colors)])
                          for i, l in enumerate(self.data.keys())])
        if markers is None:
            markers = dict([(l, default_markers[i % len(default_markers)])
                           for i, l in enumerate(self.data.keys())])
        style.update(kwargs)
        min_ = np.inf
        max_ = - np.inf
        fig = plt.figure()
        for label, results in self.data.items():
            style["color"] = colors[label]
            style["marker"] = markers[label]
            y_mean, y_std, y_num = avg_quantity(results, y, pad_y)
            y_sem = y_std / np.sqrt(y_num)
            x_mean, x_std, x_num = avg_quantity(results, x, pad_x)
            x_sem = x_std / np.sqrt(x_num)

            if xbars:
                plt.errorbar(x_mean, y_mean, xerr=x_sem, label=label,
                             ecolor="k", errorevery=xerror_every, **style)
            else:
                plt.plot(x_mean, y_mean, label=label, **style)

            if ybars:
                plt.fill_between(x_mean, y_mean - y_sem, y_mean + y_sem,
                                 alpha=.3, color=style["color"])
                max_ = max(np.max(y_mean + y_sem), max_)
                min_ = min(np.min(y_mean - y_sem), min_)
            else:
                max_ = max(y_mean.max(), max_)
                min_ = min(y_mean.min(), min_)

        # adjust visible space
        y_lim = [min_ - .1 * abs(max_ - min_), max_ + .1 * abs(max_ - min_)]
        if min_ != max_:
            plt.ylim(y_lim)

        # axis labels
        xlabel = default_labels[x] if x in default_labels else x
        ylabel = default_labels[y] if y in default_labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)

        if legend:
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0 + box.height * 0.2,
                                    box.width, box.height * 0.8])
            legend_handle = plt.legend(loc='upper center',
                                       bbox_to_anchor=(0.5, -0.15),
                                       fancybox=True, shadow=True, ncol=2)
        return fig


def save_figure(figure, filename):
    figure.savefig(
        filename,
        transparent=True,
        pad_inches=.1,
        bbox_inches='tight')
