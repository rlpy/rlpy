"""General Tools for use throughout RLPy"""


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

import sys
import numpy as np
# print "Numpy version:", numpy.__version__
# print "Python version:", sys.version_info
import os


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


__rlpy_location__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import matplotlib.backends._tkagg
    matplotlib_backend = 'tkagg'  # 'WX' 'QTAgg' 'QT4Agg'
except ImportError:
    # Anaconda is built with QT4 backend support on Windows
    matplotlib_backend = 'qt4agg'


def available_matplotlib_backends():
    def is_backend_module(fname):
        """Identifies if a filename is a matplotlib backend module"""
        return fname.startswith('backend_') and fname.endswith('.py')

    def backend_fname_formatter(fname):
        """Removes the extension of the given filename, then takes away the leading 'backend_'."""
        return os.path.splitext(fname)[0][8:]

    # get the directory where the backends live
    backends_dir = os.path.dirname(matplotlib.backends.__file__)

    # filter all files in that directory to identify all files which provide a
    # backend
    backend_fnames = filter(is_backend_module, os.listdir(backends_dir))

    backends = [backend_fname_formatter(fname) for fname in backend_fnames]
    return backends

if module_exists('matplotlib'):

    import matplotlib
    import matplotlib.backends
    import matplotlib.pyplot as plt
    mpl_backends = available_matplotlib_backends()
    if matplotlib_backend in mpl_backends:
        plt.switch_backend(matplotlib_backend)
    else:
        print "Warning: Matplotlib backend", matplotlib_backend, "not available"
        print "Available backends:", mpl_backends
    from matplotlib import pylab as pl
    import matplotlib.ticker as ticker
    from matplotlib import rc, colors
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    import matplotlib.cm as cm
    from matplotlib import lines
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import lines  # for plotting lines in pendulum and PST
    from matplotlib.mlab import rk4  # for integration in pendulum
    from matplotlib.patches import ConnectionStyle  # for cartpole
    pl.ion()
else:
    print 'matplotlib is not available => No Graphics'

if module_exists('networkx'):
    import networkx as nx
else:
    'networkx is not available => No Graphics on SystemAdmin domain'

if module_exists('sklearn'):
    from sklearn import svm
else:
    'sklearn is not available => No BEBF representation available'
from scipy import stats
from scipy import misc
from scipy import linalg
from scipy.sparse import linalg as slinalg
from scipy import sparse as sp
from time import clock
from hashlib import sha1
import datetime
import csv
from string import lower
#from Sets import ImmutableSet
#from heapq import *
import multiprocessing
from os import path
from decimal import Decimal
# If running on an older version of numpy, check to make sure we have
# defined all required functions.
import numpy as np  # We need to be able to reference numpy by name
from select import select
from itertools import combinations, chain

def discrete_sample(p):
    cp = np.cumsum(p)
    return np.sum(cp <= np.random.rand(1))


def matrix_mult(A, B):
    # TO BE COMPLETED!
    # This function is defined due to many frustration with the dot, *
    # operators that behave differently based on the input values:
    # sparse.matrix, matrix, ndarray and array
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    if len(B.shape) == 1:
        B = B.reshape(1, -1)
    n1, m1 = A.shape
    n2, m2 = B.shape
    if m1 != n2:
        print "Incompatible dimensions: %dx%d and %dx%d" % (n1, m2, n2, m2)
        return None
    else:
        return A.dot(B)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

# if numpy.version.version < '2.6.0': # Missing count_nonzero


def count_nonzero(arr):
    # NOTE that the count_nonzero function below moves recursively through any sublists,
    # such that only individual elements are examined.
    # Some versions of numpy's count_nonzero only strictly compare each element;
    # e.g. numpy.count_nonzero([[1,2,3,4,5], [6,7,8,9]]) might return 2, while
    # Tools.count_nonzero([[1,2,3,4,5], [6,7,8,9]]) returns 9.
    # The latter is the desired functionality, irrelevant when single
    # arrays/lists are passed.
    nnz = 0

    # Is this an instance of a matrix? Use inbuilt nonzero() method and count # of indices returned.
    # NOT TESTED with high-dimensional matrices (only 2-dimensional matrices)
    if sp.issparse(arr):
        return arr.getnnz()

    if isinstance(arr, np.matrixlib.defmatrix.matrix):
        # Tuple of length = # dimensions (usu. 2) containing indices of nonzero
        # elements
        nonzero_indices = arr.nonzero()
        # Find # of indices in the vector corresponding to any of the
        # dimensions (all have same length)
        nnz = np.size(nonzero_indices[0])
        return nnz

    if isinstance(arr, np.ndarray):
        # return sum([1 for x in arr.ravel() if x != 0])
        return np.count_nonzero(arr.ravel())

    if isinstance(arr, list):
        for el in arr:
            if isinstance(el, list):
                nnz += np.count_nonzero(el)
            elif el != 0:
                nnz += 1
        return nnz

    print "In tools.py attempted count_nonzero with unsupported type of", type(arr)
    return None

# Tips:
# array.astype(float) => convert elements
# matlibplot initializes the maping from the values to
# colors on the first time creating unless bounds are set manually.
# Hence you may update color values later but dont see any updates!
# in specifying dimensions for reshape you can put -1 so it will be automatically infered
# [2,2,2] = [2]*3
# [1,2,2,1,2,2,1,2,2] = ([1]+[2]*2)*3
# [[1,2],[1,2],[1,2]] = array([[1,2],]*3)
# apply function foo to all elements of array A: vectorize(foo)(A) (The operation may be unstable! Care!
# Set a property of a class:  vars(self)['prop'] = 2
# dont use a=b=zeros((2,3)) because a and b will point to the same array!
# b = A[:,1] does NOT create a new matrix. It is simply a pointer to that row! so if you change b you change A
# DO NOT USE A = B = array() unless you know what you are doing. They will point to the same object!
# Todo:
# Replace vstack and hstack with the trick mentioned here:
# http://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array
# if undo redo does not work in eclipse, you may have an uninfinished
# process. Kill all


def randint(low, high, m=1, n=1):
    return np.random.randint(low, high + 1, size=(m, n))


def randSet(x):
    # Returns a random element of a list uniformly.
    #i = random.random_integers(0,size(x)-1)
    i = np.random.randint(0, len(x) - 1)
    return x[i]


def closestDiscretization(x, bins, limits):
    # Return the closest point to x based on the discretization defined by the number of bins and limits
    # equivalent to state2bin(x) / (bins-1) * width + limits[0]
    #width = limits[1]-limits[0]
    # return round((x-limits[0])*bins/(width*1.)) / bins * width + limits[0]
    return bin2state(state2bin(x, bins, limits), bins, limits)


def bin2state(bin, bins, limits):
    # inverse of state2bin function
    # Given a bin number and the number of the bins and the limits on a single
    # dimension it return the corresponding value in the middle of the bin
    bin_width = (limits[1] - limits[0]) / (bins * 1.)
    return bin * bin_width + bin_width / 2.0 + limits[0]


def state2bin(s, bins, limits):
    # return the bin number corresponding to state s given a state it returns a vector with the same dimensionality of s
    # note that s can be continuous.
    # examples:
    # s = 0, limits = [-1,5], bins = 6 => 1
    # s = .001, limits = [-1,5], bins = 6 => 1
    # s = .4, limits = [-.5,.5], bins = 3 => 2
    # each element of the returned valued is the zero-indexed bin number
    # corresponding to s
    if s == limits[1]:
        return bins - 1
    width = limits[1] - limits[0]
    if s > limits[1]:
        print 'Tools.py: WARNING: ', s, ' > ', limits[1], '. Using the chopped value of s'
        print 'Ignoring', limits[1] - s
        s = limits[1]
    elif s < limits[0]:
        print 'Tools.py: WARNING: ', s, ' < ', limits[0], '. Using the chopped value of s'
#        print("WARNING: %s is out of limits of %s . Using the chopped value of s" %(str(s),str(limits)))
        s = limits[0]
    return int((s - limits[0]) * bins / (width * 1.))


def deltaT(start_time):
    return clock() - start_time


def hhmmss(t):
    # Return a string of hhmmss
    return str(datetime.timedelta(seconds=round(t)))


def className(obj):
    # return the name of a class
    return obj.__class__.__name__


def createColorMaps():
    # Make Grid World ColorMap
    mycmap = colors.ListedColormap(
        ['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(['r', 'k'], 'fiftyChainActions')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(['b', 'r'], 'FlipBoard')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(
        ['w', '.75', 'b', 'r'], 'IntruderMonitoring')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(
        ['w', 'b', 'g', 'r', 'm', (1, 1, 0), 'k'], 'BlocksWorld')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(['.5', 'k'], 'Actions')
    cm.register_cmap(cmap=mycmap)
    # mycmap = make_colormap({0:(.8,.7,0), 1: 'w', 2:(0,0,1)})  # orange to
    # blue
    mycmap = make_colormap({0: 'r', 1: 'w', 2: 'g'})  # red to blue
    cm.register_cmap(cmap=mycmap, name='ValueFunction')
    mycmap = colors.ListedColormap(['r', 'w', 'k'], 'InvertedPendulumActions')
    cm.register_cmap(cmap=mycmap)

    mycmap = colors.ListedColormap(['r', 'w', 'k'], 'MountainCarActions')
    cm.register_cmap(cmap=mycmap)
    mycmap = colors.ListedColormap(['r', 'w', 'k', 'b'], '4Actions')
    cm.register_cmap(cmap=mycmap)


def make_colormap(colors):
    """
    Define a new color map based on values specified in the dictionary
    colors, where colors[z] is the color that value z should be mapped to,
    with linear interpolation between the given values of z.

    The z values (dictionary keys) are real numbers and the values
    colors[z] can be either an RGB list, e.g. [1,0,0] for red, or an
    html hex string, e.g. "#ff0000" for red.
    """

    from matplotlib.colors import LinearSegmentedColormap, ColorConverter

    z = np.sort(colors.keys())
    n = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / ((zn - z1) * 1.)

    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in xrange(n):
        # i'th color at level z[i]:
        Ci = colors[z[i]]
        if isinstance(Ci, str):
            # a hex string of form '#ff0000' for example (for red)
            RGB = CC.to_rgb(Ci)
        else:
            # assume it's an RGB triple already:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red'] = [(x0[i], R[i], R[i]) for i in xrange(len(R))]
    cmap_dict['green'] = [(x0[i], G[i], G[i]) for i in xrange(len(G))]
    cmap_dict['blue'] = [(x0[i], B[i], B[i]) for i in xrange(len(B))]
    mymap = LinearSegmentedColormap('mymap', cmap_dict)
    return mymap


def showcolors(cmap):
    plt.clf()
    x = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, x)
    plt.pcolor(X, Y, 0.5 * (X + Y), cmap=cmap, edgecolors='k')
    plt.axis('equal')
    plt.colorbar()
    plt.title('Plot of x+y using colormap')


def schlieren_colormap(color=[0, 0, 0]):
    """
    For Schlieren plots:
    """
    if color == 'k':
        color = [0, 0, 0]
    if color == 'r':
        color = [1, 0, 0]
    if color == 'b':
        color = [0, 0, 1]
    if color == 'g':
        color = [0, 0.5, 0]
    if color == 'y':
        color = [1, 1, 0]
    color = np.array([1, 1, 1]) - np.array(color)
    s = np.linspace(0, 1, 20)
    colors = {}
    for key in s:
        colors[key] = np.array([1, 1, 1]) - key ** 10 * color
    schlieren_colors = make_colormap(colors)
    return schlieren_colors


def make_amrcolors(nlevels=4):
    """
    Make lists of colors useful for distinguishing different grids when
    plotting AMR results.

    INPUT::
       nlevels: maximum number of AMR levels expected.
    OUTPUT::
       (linecolors, bgcolors)
       linecolors = list of nlevels colors for grid lines, contour lines
       bgcolors = list of nlevels pale colors for grid background
    """

    # For 4 or less levels:
    linecolors = ['k', 'b', 'r', 'g']
    # Set bgcolors to white, then light shades of blue, red, green:
    bgcolors = ['#ffffff', '#ddddff', '#ffdddd', '#ddffdd']
    # Set bgcolors to light shades of yellow, blue, red, green:
    # bgcolors = ['#ffffdd','#ddddff','#ffdddd','#ddffdd']

    if nlevels > 4:
        linecolors = 4 * linecolors  # now has length 16
        bgcolors = 4 * bgcolors
    if nlevels <= 16:
        linecolors = linecolors[:nlevels]
        bgcolors = bgcolors[:nlevels]
    else:
        print "*** Warning, suggest nlevels <= 16"

    return (linecolors, bgcolors)


def linearMap(x, a, b, A=0, B=1):
    # This function takes scalar X in range [a1,b1] and maps it to [A1,B1]
    # values oout of a and b are clipped to boundaries
    if a == b:
        res = B
    else:
        res = (x - a) / (1. * (b - a)) * (B - A) + A
    if res < A:
        res = A
    if res > B:
        res = B
    return res


def generalDot(x, y):
    if sp.issparse(x):
        #active_indices = x.nonzero()[0].flatten()
        return x.multiply(y).sum()
    else:
        return np.dot(x, y)


def normpdf(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)


def factorial(x):
    return misc.factorial(x)


def nchoosek(n, k):
    return misc.comb(n, k)


def findElem(x, A):
    if x in A:
        return A.index(x)
    else:
        return []


def findElemArray1D(x, A):  # Returns an array of indices in x where x[i] == A
    res = np.where(A == x)
    if len(res[0]):
        return res[0].flatten()
    else:
        return []


def findElemArray2D(x, A):
    # Find the index of element x in array A
    res = np.where(A == x)
    if len(res[0]):
        return res[0].flatten(), res[1].flatten()
    else:
        return [], []


def findRow(r, X):
    # return the indices of X that are equal to r.
    # r and X must have the same number of columns
    # return nonzero(any(logical_and.reduce([X[:, i] == r[i] for i in arange(len(r))])))
    # return any(logical_and(X[:, 0] == r[0], X[:, 1] == r[1]))
    ind = np.nonzero(np.logical_and.reduce([X[:, i] == r[i] for i in xrange(len(r))]))
    return ind[0]




def perms(X):
    # Returns all permutations
    # X = [2 3]
    # res = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]
    # X = [[1,3],[2,3]]
    # res = [[1,2],[1,3],[3,2],[3,3]
    # Outputs are in numpy array format
    allPerms, _ = perms_r(X, perm_sample=np.array([]), allPerms=None, ind=0)

    return allPerms
######################################################


def perms_r(X, perm_sample=np.array([]), allPerms=None, ind=0):
    if allPerms is None:
        # Get memory
        if isinstance(X[0], list):
            size = np.prod([len(x) for x in X])
        else:
            size = np.prod(X, dtype=np.int)
        allPerms = np.zeros((size, len(X)))
    if len(X) == 0:
        allPerms[ind, :] = perm_sample
        perm_sample = np.array([])
        ind = ind + 1
    else:
        if isinstance(X[0], list):
            for x in X[0]:
                allPerms, ind = perms_r(
                    X[1:], np.hstack((perm_sample, [x])), allPerms, ind)
        else:
            for x in xrange(X[0]):
                allPerms, ind = perms_r(
                    X[1:], np.hstack((perm_sample, [x])), allPerms, ind)
    return allPerms, ind
######################################################


def vec2id2(x, limits):
    # returns a unique id by calculating the enumerated number corresponding to a vector given the number of bins in each dimension
    # Slower than the other implementation by a factor of 2
    if isinstance(x, int):
        return x
    lim_prod = np.cumprod(limits[:-1])
    return x[0] + sum(map(lambda x_y: x_y[0] * x_y[1], zip(x[1:], lim_prod)))


def vec2id(x, limits):
    # returns a unique id by calculating the enumerated number corresponding to a vector given the number of bins in each dimension
    # I use a recursive calculation to save time by looping once backward on the array = O(n)
    # for example:
    # vec2id([0,1],[5,10]) = 5
    if isinstance(x, int):
        return x
    _id = 0
    for d in xrange(len(x) - 1, -1, -1):
        _id *= limits[d]
        _id += x[d]

    return _id
######################################################


def id2vec(_id, limits):
    # returns the vector corresponding to an id given the number of buckets in each dimension (invers of vec2id)
    # for example:
    # id2vec(5,[5,10]) = [0,1]
    prods = np.cumprod(limits)
    s = [0] * len(limits)
    for d in xrange(len(prods) - 1, 0, -1):
#       s[d] = _id / prods[d-1]
#       _id %= prods[d-1]
        s[d], _id = divmod(_id, prods[d - 1])
    s[0] = _id
    return s


def bound_vec(X, limits):
    # Input:
    # given X as 1-by-n array
    # limits as 2-by-n array
    # Output:
    # array where each element is bounded between the corresponding bounds in the limits
    # i.e limits[i,0] <= output[i] <= limits[i,1]
    MIN = limits[:, 0]
    MAX = limits[:, 1]
    X = np.vstack((X, MIN))
    X = np.amax(X, axis=0)
    X = np.vstack((X, MAX))
    X = np.amin(X, axis=0)
    return X


def bound(x, m, M=None):
    # x should all be scalar
    # m can be scalar or a [min,max] format
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def wrap(x, m, M):
    # wrap m between min (m) and Max (M)
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def powerset(iterable, ascending=1):
    s = list(iterable)
    if ascending:
        return (
            chain.from_iterable(combinations(s, r) for r in xrange(len(s) + 1))
        )
    else:
        return (
            chain.from_iterable(combinations(s, r)
                                for r in xrange(len(s) + 1, -1, -1))
        )


def printClass(obj):
    print className(obj)
    print '======================================='
    for property, value in vars(obj).iteritems():
        print property, ": ", value


def normalize(x):
    # normalize numpy array x
    return x / sum([e ** 2 for e in x])


def addNewElementForAllActions(x, a, newElem=None):
    # When features are expanded several parameters such as the weight vector should expand. Since we are adding the new feature for all actions
    # these vectors should expand by size of the action as for each action phi(s) is expand by 1 element.
    # Because we later stack all of them we need this function to insert the new element in proper locations
    # Add a new 0 weight corresponding to the new added feature for all actions.
    # new elem = None means just insert zeros for new elements. x is a numpy array
    # example:
    # x = [1,2,3,4], a = 2, newElem = None => [1,2,0,3,4,0]
    # x = [1,2,3], a = 3, newElem = [1,1,1] => [1,1,2,1,3,1]
    if newElem is None:
        newElem = np.zeros((a, 1))
    if len(x) == 0:
        return newElem.flatten()
    else:
        x = x.reshape(a, -1)  # -1 means figure the other dimension yourself
        x = np.hstack((x, newElem))
        x = x.reshape(1, -1).flatten()
        return x


def solveLinear(A, b):
    # Solve the linear equation Ax=b.
    # return x and the time for solve
    error = np.inf  # just to be safe, initialize error variable here
    if sp.issparse(A):
    # print 'sparse', type(A)
        start_log_time = clock()
        result = slinalg.spsolve(A, b)
        solve_time = deltaT(start_log_time)
        error = linalg.norm((A * result.reshape(-1, 1) - b.reshape(-1, 1))[0])
        # For extensive comparision of methods refer to InversionComparison.txt
    else:
        # print 'not sparse, type',type(A)
        if sp.issparse(A):
            A = A.todense()
        # Regularize A
        # result = linalg.lstsq(A,b); result = result[0] # Extract just the
        # answer
        start_log_time = clock()
        result = linalg.solve(A, b)
        solve_time = deltaT(start_log_time)

        # use numpy matrix multiplication
        if isinstance(A, np.matrixlib.defmatrix.matrix):
            error = np.linalg.norm(
                (A * result.reshape(-1, 1) - b.reshape(-1, 1))[0])
        elif isinstance(A, np.ndarray):  # use array multiplication
            error = np.linalg.norm(
                (np.dot(A, result.reshape(-1, 1)) - b.reshape(-1, 1))[0])
        else:
            print 'Attempted to solve linear equation Ax=b in solveLinear() of Tools.py with a non-numpy (array / matrix) type.'
            sys.exit(1)

    if error > RESEDUAL_THRESHOLD:
        print "||Ax-b|| = %0.1f" % error
    return result.ravel(), solve_time


def rank(A, eps=1e-12):
    u, s, v = linalg.svd(A)
    return len([x for x in s if abs(x) > eps])


def fromAtoB(x1, y1, x2, y2, color='k', connectionstyle="arc3,rad=-0.4",
             shrinkA=10, shrinkB=10, arrowstyle="fancy", ax=None):
    # draw an arrow from point A=(x1,y1) to point B=(x2,y2)
    # ax is optional to specifify the axis used for drawing
    if ax is None:
        return pl.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )
    else:
        return ax.annotate("",
                           xy=(x2, y2), xycoords='data',
                           xytext=(x1, y1), textcoords='data',
                           arrowprops=dict(
                               arrowstyle=arrowstyle,  # linestyle="dashed",
                               color=color,
                               shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None,
                               patchB=None,
                               connectionstyle=connectionstyle),
                           )


def drawHist(data, bins=50, fig=101):
    hist, bins = np.histogram(data, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.figure(fig)
    plt.bar(center, hist, align='center', width=width)


def nonZeroIndex(A):
    # Given a 1D array it returns the list of non-zero index of the Array
    # [0,0,0,1] => [4]
    return A.nonzero()[0]


def sp_matrix(m, n=1, dtype='float'):
    # returns a sparse matrix with m rows and n columns, with the dtype
    # We use dok_matrix for sparse matrixies
    return sp.csr_matrix((m, n), dtype=dtype)


def sp_dot_array(sp_m, A):
    # Efficient dot product of matrix sp_m in shape of 1-by-p and array A with
    # p elements
    assert sp_m.shape[1] == len(A)
    ind = sp_m.nonzero()[1]
    if len(ind) == 0:
        return 0
    if sp_m.dtype == 'bool':
        # Just sum the corresponding indexes of theta
        return sum(A[ind])
    else:
        # Multiply by feature values since they are not binary

        return sum([A[i] * sp_m[0, i] for i in ind])


def sp_dot_sp(sp_1, sp_2):
    # Efficient dot product of matrix sp_m in shape of p-by-1 and array A with
    # p elements
    assert sp_1.shape[
        0] == sp_2.shape[
        0] and sp_1.shape[
        1] == 1 and sp_2.shape[
        1] == 1
    ind_1 = sp_1.nonzero()[0]
    ind_2 = sp_2.nonzero()[0]
    if len(ind_1) * len(ind_2) == 0:
        return 0

    ind = np.intersect1d(ind_1, ind_2)
    # See if they are boolean
    if sp_1.dtype == bool and sp_2.dtype == bool:
        return len(ind)
    sp_bool = None
    if sp_1.dtype == bool:
        sp_bool = sp_1
        sp = sp_2
    if sp_2.dtype == bool:
        sp_bool = sp_2
        sp = sp_1
    if sp_bool is None:
        # Multiply by feature values since they are not binary
        return sum([sp_1[i, 0] * sp_2[i, 0] for i in ind])
    else:
        return sum([sp[i, 0] for i in ind])


def sp_add2_array(sp, A):
    # sp is a sparse matrix p-by-1
    # A is an array of len p
    # this function return an array corresponding to A+sp
    ind = sp.nonzero()[0]
    for i in ind:
        A[i] += sp[i, 0]
    return A


def checkNCreateDirectory(fullfilename):
    # See if a fullfilename exists if not create the required directory

    path_, _, _ = fullfilename.rpartition('/')
    if not os.path.exists(path_):
        os.makedirs(path_)


def hasFunction(object, methodname):
    method = getattr(object, methodname, None)
    return callable(method)


def pretty(X, format='%0.3f'):
    # convert a numpy array in given format to str
    # [1,2,3], %0.3f => 1.000    2.000    3.000
    format = format + '\t'
    return ''.join(format % x for x in X)


def regularize(A):
    # Adds REGULARIZATION*I To A. This is often done before calling the linearSolver
    # A has to be square matrix
    x, y = A.shape
    assert x == y  # Square matrix
    if sp.issparse(A):
        A = A + REGULARIZATION * sp.eye(x, x)
        # print 'REGULARIZE', type(A)
    else:
        # print 'REGULARIZE', type(A)
        for i in xrange(x):
            A[i, i] += REGULARIZATION
    return A


def sparsity(A):
    # Returns the percent [0-100] of elements of A that are 0
    return (1 - np.count_nonzero(A) / (np.prod(A.shape) * 1.)) * 100


def printMatrix(A, type='int'):
    # print a matrix in a desired format
    print np.array(A, dtype=type)


def incrementalAverageUpdate(avg, sample, sample_number):
    return avg + (sample - avg) / (sample_number * 1.)


def padZeros(X, L):
    # X is a 1D numpy array
    # L is an int
    # if len(X) < L pad zeros to X so it will have length L
    if len(X) < L:
        new_X = np.zeros(L)
        new_X[:len(X)] = X
        return new_X
    else:
        return X


def expectedPhiNS(p_vec, ns_vec, representation):
    # Primarily for use with domain.expectedStep()
    # Takes p_vec, probability of each state outcome in ns_vec,
    # Returns a vector of length features_num which is the expectation
    # over all possible outcomes.
    expPhiNS = np.zeros(representation.features_num)
    for i, ns in enumerate(ns_vec):
        expPhiNS += p_vec[i] * representation.phi_nonTerminal(ns)
    return expPhiNS
    #  p: k-by-1    probability of each transition
    #  r: k-by-1    rewards
    # ns: k-by-|s|  next state
    #  t: k-by-1    terminal values


def allExpectedPhiNS(domain, representation, policy, allStates=None):
    # Returns Phi' matrix with dimensions n x k,
    # n: number of possible states, and
    # k: number of features
    if allStates is None:
        allStates = domain.allStates()
    allExpPhiNS = np.zeros((len(allStates), representation.features_num))
    for i, s in enumerate(allStates):
#         print s
#         print policy.pi(s)
#         print 'looping',i, policy.pi(s)
#         print policy.pi(s)
        p_vec, r_vec, ns_vec, t_vec = domain.expectedStep(s, policy.pi(s))
        allExpPhiNS[i][:] = expectedPhiNS(p_vec, ns_vec, representation)
    return allExpPhiNS


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


# Setup the latdex path
# if sys.platform == 'darwin':
    #os.environ['PATH'] += ':' + TEXPATH
# if sys.platform == 'win32':
#    print os.environ['PATH']
    #os.environ['PATH'] += ';' + TEXPATH

# def isLatexConfigured():
#    return False
#    try:
#        pl.subplot(1,3,2)
#        pl.xlabel(r"$\theta$")
#        pl.show()
#        pl.draw()
#        pl.close()
#        print "Latex tested and functioning"
#    except:
#        print "Matplotlib failed to plot, likely due to a Latex problem."
#        print "Check that your TEXPATH is set correctly in config.py,"
#        print "and that latex is installed correctly."
# print "\nDisabling latex functionality, using matplotlib native fonts."

if module_exists('matplotlib'):
    createColorMaps()
    rc('font', family='serif', size=15,
       weight="bold", **{"sans-serif": ["Helvetica"]})
    rc("axes", labelsize=15)
    rc("xtick", labelsize=15)
    rc("ytick", labelsize=15)
    # rc('text',usetex=False)

    # Try to use latex fonts, if available
    # rc('text',usetex=True)

# Colors
PURPLE = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
NOCOLOR = '\033[0m'
RESEDUAL_THRESHOLD = 1e-7
REGULARIZATION = 1e-6
FONTSIZE = 15
SEP_LINE = "=" * 60
