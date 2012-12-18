######################################################
# Developed by Alborz Geramiard Oct 26th 2012 at MIT #
######################################################

from operator import *
from numpy  import *
#matplotlib.use("WXAgg") # do this before pylab so you don'tget the default back end. < Maybe faster but I dont have the package yet
from matplotlib import pylab as pl
pl.ion()
from matplotlib import mpl,rc
import glob
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as col
import matplotlib.cm as cm
from scipy import stats
from scipy import misc
from scipy import linalg
from scipy.sparse import linalg as slinalg
from scipy import sparse as sp 
from time import *
from hashlib import sha1
import datetime, csv
from string import *
from sets import ImmutableSet,Set
from itertools  import *
from copy import deepcopy
import networkx as nx
import os 
import sys
from os import path
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
# if undo redo does not work in eclipse, you may have an uninfinished process. Kill all
def randSet(x):
    #Returns a random element of a list uniformly.
    i = random.random_integers(0,size(x)-1)
    return x[i]
def closestDiscretization(x, bins, limits):
    #Return the closest point to x based on the discretization defined by the number of bins and limits
    # equivalent to binNumber(x) / (bins-1) * width + limits[0]
    width = limits[1]-limits[0]
    return round((x-limits[0])*bins/(width*1.)) / bins * width + limits[0]
def binNumber(s,bins,limits):
    # return the bin number corresponding to s given Given a state it returns a vector with the same dimensionality of s
    # each element of the returned valued is the zero-indexed bin number corresponding to s
    # note that s can be continuous.  
    # 1D examples: 
    # s = 0, limits = [-1,5], bins = 6 => 1
    # s = .001, limits = [-1,5], bins = 6 => 1
    # s = .4, limits = [-.5,.5], bins = 3 => 2
    if s == limits[1]: 
        return bins-1
    width = limits[1]-limits[0]
    if s > limits[1]:
        print 'Tools.py: WARNING: ',s,' > ',limits[1],'. Using the chopped value of s'
        print 'Ignoring', limits[1] - s
        s = limits[1]
    elif s < limits[0]:
        print 'Tools.py: WARNING: ',s,' < ',limits[0],'. Using the chopped value of s'
#        print("WARNING: %s is out of limits of %s . Using the chopped value of s" %(str(s),str(limits)))
        s = limits[0]
    return int((s-limits[0])*bins/(width*1.))
def deltaT(start_time):
    return time()-start_time
def hhmmss(t):
    #Return a string of hhmmss
    return str(datetime.timedelta(seconds=round(t)))
def className(obj):
    # return the name of a class
    return obj.__class__.__name__
def scale(x,m,M):
    # given an array return the scaled version of the array.
    # positive numbers are scaled [0,M] -> [0,1]
    # negative numbers are scaled [m,0] -> [-1,0]
    pos_ind = where(x>0)
    #x(pos_ind) = x(pos_ind)
def createColorMaps():
    #Make Grid World ColorMap
    mycmap = col.ListedColormap(['w', '.75','b','g','r','k'], 'GridWorld')
    cm.register_cmap(cmap=mycmap)
    mycmap = col.ListedColormap(['w','.75','b','r'], 'IntruderMonitorying')
    cm.register_cmap(cmap=mycmap)
    mycmap = col.ListedColormap(['w','b','g','r','m',(1,1,0),'k'], 'BlocksWorld')
    cm.register_cmap(cmap=mycmap)
    mycmap = col.ListedColormap(['.6','k'], 'Actions')
    cm.register_cmap(cmap=mycmap)
    mycmap = make_colormap({0:'r', 1: 'w', 2.:'g'})  # red to blue
    cm.register_cmap(cmap=mycmap,name='ValueFunction')
    mycmap = col.ListedColormap(['r','w','k'], 'InvertedPendulumActions')
    cm.register_cmap(cmap=mycmap)
#    Some useful Colormaps
#    red_yellow_blue = make_colormap({0.:'r', 0.5:'#ffff00', 1.:'b'})
#    blue_yellow_red = make_colormap({0.:'b', 0.5:'#ffff00', 1.:'r'})
#    yellow_red_blue = make_colormap({0.:'#ffff00', 0.5:'r', 1.:'b'})
#    white_red = make_colormap({0.:'w', 1.:'r'})
#    white_blue = make_colormap({0.:'w', 1.:'b'})
#    
#    schlieren_grays = schlieren_colormap('k')
#    schlieren_reds = schlieren_colormap('r')
#    schlieren_blues = schlieren_colormap('b')
#    schlieren_greens = schlieren_colormap('g')
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
    from numpy import sort
    
    z = sort(colors.keys())
    n = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / ((zn - z1)*1.)
    
    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        #i'th color at level z[i]:
        Ci = colors[z[i]]      
        if type(Ci) == str:
            # a hex string of form '#ff0000' for example (for red)
            RGB = CC.to_rgb(Ci)
        else:
            # assume it's an RGB triple already:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red'] = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    return mymap
def showcolors(cmap):
    from pylab import colorbar, clf, axes, linspace, pcolor, \
         meshgrid, show, axis, title
    #from scitools.easyviz.matplotlib_ import colorbar, clf, axes, linspace,\
                 #pcolor, meshgrid, show, colormap
    clf()
    x = linspace(0,1,21)
    X,Y = meshgrid(x,x)
    pcolor(X,Y,0.5*(X+Y), cmap=cmap, edgecolors='k')
    axis('equal')
    colorbar()
    title('Plot of x+y using colormap')
def schlieren_colormap(color=[0,0,0]):
    """
    For Schlieren plots:
    """
    from numpy import linspace, array
    if color=='k': color = [0,0,0]
    if color=='r': color = [1,0,0]
    if color=='b': color = [0,0,1]
    if color=='g': color = [0,0.5,0]
    if color=='y': color = [1,1,0]
    color = array([1,1,1]) - array(color)
    s  = linspace(0,1,20)
    colors = {}
    for key in s:
        colors[key] = array([1,1,1]) - key**10 * color
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
    bgcolors = ['#ffffff','#ddddff','#ffdddd','#ddffdd']
    # Set bgcolors to light shades of yellow, blue, red, green:
    #bgcolors = ['#ffffdd','#ddddff','#ffdddd','#ddffdd']

    if nlevels > 4:
        linecolors = 4*linecolors  # now has length 16
        bgcolors = 4*bgcolors
    if nlevels <= 16:
        linecolors = linecolors[:nlevels]
        bgcolors = bgcolors[:nlevels]
    else:
        print "*** Warning, suggest nlevels <= 16"

    return (linecolors, bgcolors)
def linearMap(x,a,b,A=0,B=1):
    # This function takes scalar X in range [a1,b1] and maps it to [A1,B1]
    # values oout of a and b are clipped to boundaries 
    if a == b:
        res = B
    else:
        res = (x-a)/(1.*(b-a))*(B-A)+A
    if res < A: res = A
    if res > B: res = B
    return res
def isSparse(x):
    return isinstance(x,sparse.lil_matrix) or isinstance(x,sparse.csr_matrix)        
def generalDot(x,y):
    if isSparse(x):
        #active_indecies = x.nonzero()[0].flatten()
        return x.multiply(y).sum()
    else:
        return dot(x,y)
def normpdf(x, mu, sigma):
    return stats.norm.pdf(x,mu,sigma)
def factorial(x):
    return misc.factorial(x)
def nchoosek(n,k):
    return misc.comb(n,k)
def findElem(x,A):
    if x in A:
        return A.index(x)
    else:
        return []    
def findElemArray1D(x,A): # Returns an array of indices in x where x[i] == A
    res = where(A==x)
    if len(res[0]):
        return res[0].flatten()
    else:
        return []
def findElemArray2D(x,A):
    # Find the index of element x in array A
    res = where(A==x)
    if len(res[0]):
        return res[0].flatten(), res[1].flatten()
    else:
        return [], []
def findRow(r,X):
    # return the indices of X that are equal to X.
    # row and X must have the same number of columns
    #return nonzero(any(logical_and.reduce([X[:, i] == r[i] for i in range(len(r))])))
    #return any(logical_and(X[:, 0] == r[0], X[:, 1] == r[1]))
    ind = nonzero(logical_and.reduce([X[:, i] == r[i] for i in range(len(r))]))
    return ind[0] 
def perms(X):
    # Returns all permutations
    # X = [2 3]
    # res = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]
    # X = [[1,3],[2,3]]
    # res = [[1,2],[1,3],[3,2],[3,3]
    # Outputs are in numpy array format
    allPerms, _ = perms_r(X, perm_sample= array([],'uint8') , allPerms = None,ind = 0)
    return allPerms
######################################################
def perms_r(X, perm_sample= array([],'uint8') , allPerms = None,ind = 0):
    if allPerms is None:
        #Get memory
        if isinstance(X[0], list):
            size        = prod([len(x) for x in X])
        else:
            size        = prod(X)
        allPerms    = zeros((size,len(X)),'uint8')
    if len(X) == 0:
        allPerms[ind,:] = perm_sample
        perm_sample = array([],'uint8')
        ind = ind + 1;
    else:
        if isinstance(X[0], list):
            for x in X[0]:
                allPerms, ind = perms_r(X[1:],hstack((perm_sample, [x])), allPerms, ind)
        else:
            for x in arange(X[0]):
                allPerms, ind = perms_r(X[1:],hstack((perm_sample, [x])), allPerms, ind)
    return allPerms, ind
######################################################
def vec2id2(x,limits):
    #returns a unique id by calculating the enumerated number corresponding to a vector given the limits on each dimenson of the vector
    # I use a recursive calculation to save time by looping once backward on the array = O(n)
    # Slower than the other implementation by a factor of 2
    if isinstance(x,int): return x 
    lim_prod = cumprod(limits[:-1])
    return x[0] + sum(map(lambda (x,y):x*y,zip(x[1:],lim_prod))) 
def vec2id(x,limits):
    #returns a unique id by calculating the enumerated number corresponding to a vector given the limits on each dimenson of the vector
    # I use a recursive calculation to save time by looping once backward on the array = O(n)
    if isinstance(x,int): return x 
    _id = 0
    for d in arange(len(x)-1,-1,-1):
        _id *= limits[d]
        _id += x[d]
    
    return _id
######################################################
def id2vec(_id,limits):
    #returns the vector corresponding to an id given the limits (invers of vec2id)
    prods = cumprod(limits)
    s = [0] * len(limits)
    for d in arange(len(prods)-1,0,-1):
#       s[d] = _id / prods[d-1]
#       _id %= prods[d-1]
        s[d], _id = divmod(_id, prods[d-1])
    s[0] = _id
    return s
def bound(x,m,M):
    # bound x between min (m) and Max (M)
    return min(max(x,m),M)
def wrap(x,m,M):
    # wrap m between min (m) and Max (M)
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x
def shout(obj,s=''):
    # Print the name of the object and then the message. Use to remember to comment prints
    print "In " + className(obj) + " :" + str(s) 
def powerset(iterable, ascending = 1):
    s = list(iterable)
    if ascending:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1,-1,-1))
def printClass(obj):
        print className(obj)
        print '======================================='
        for property, value in vars(obj).iteritems():
            print property, ": ", value
def normalize(x):
    # normalize numpy array x
    return x/sum([e**2 for e in x])
def addNewElementForAllActions(x,a,newElem = None):
        # When features are expanded several parameters such as the weight vector should expand. Since we are adding the new feature for all actions
        # these vectors should expand by size of the action as for each action phi(s) is expand by 1 element.
        # Because we later stack all of them we need this function to insert the new element in proper locations 
        # Add a new 0 weight corresponding to the new added feature for all actions.
        # new elem = None means just insert zeros for new elements. x is a numpy array
        # example:
        # x = [1,2,3,4], a = 2, newElem = None => [1,2,0,3,4,0]
        # x = [1,2,3], a = 3, newElem = [1,1,1] => [1,1,2,1,3,1]
        if newElem is None:
            newElem = zeros((a,1))
        if len(x) == 0:
            return newElem.flatten()
        else:
            x   = x.reshape(a,-1) # -1 means figure the other dimension yourself
            x   = hstack((x,newElem))
            x   = x.reshape(1,-1).flatten()
            return x
def solveLinear(A,b):
    # Solve the linear equation Ax=b.
    if sp.issparse(A) and False:
        result = slinalg.lsmr(A,b)
        # Timing in seconds, for solving 100x100 with 100 elements in A for Ax=b problems
        #def foo():
        #    L = 100
        #    o = ones(L)
        #    rows = random.random_integers(0,L-1,L)
        #    cols = random.random_integers(0,L-1,L)
        #    M = sp.csc_matrix((o,(rows,cols)),shape=(L,L))
        #    b = arange(L)
        #    x = slinalg.lsmr(M,b)
        #
        #random.seed(999)
        #t = timeit.Timer(stmt="foo()",setup="from __main__ import *")
        #print t.timeit(number=100)          
        # lsmr  = .54 (s)
        # lsqr  = .58 (s)
        # cg    = 3.5 (s)
        # bicg  = 6.8 (s)
        # qmr   = 9.5 (s)
    else:
        result = linalg.lstsq(A.todense(),b)
    return result[0]
def rows(A):
    # return the rows of matrix A
    r, c = A.shape
    return r
def cols(A):
    # return the rows of matrix A
    r, c = A.shape
    return c
def rank(A, eps=1e-12):
    u, s, v = linalg.svd(A)
    return len([x for x in s if abs(x) > eps])
def easy2read(A, _precision=3):
    # returns an array easy to read (used for debugging mainly. _precision is the number of decimal digits
    return array_repr(A, precision=_precision, suppress_small=True)
def fromAtoB(x1,y1,x2,y2,color = 'k', connectionstyle="arc3,rad=-0.4",shrinkA=10,shrinkB=10,arrowstyle="fancy"):
    #draw an arrow from point A=(x1,y1) to point B=(x2,y2)
    return pl.annotate("",
                xy=(x2,y2), xycoords='data',
                xytext=(x1,y1), textcoords='data',
                arrowprops=dict(arrowstyle=arrowstyle, #linestyle="dashed",
                                color= color,
                                shrinkA=shrinkA, shrinkB=shrinkB,
                                patchA=None,
                                patchB=None,
                                connectionstyle=connectionstyle), 
                )
def drawHist(data,bins=50,fig=101):
    hist, bins = histogram(data,bins = bins)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    pl.figure(fig)
    pl.bar(center, hist, align = 'center', width = width)
def nonZeroIndex(A):
    # Given a 1D array it returns the list of non-zero index of the Array
    # [0,0,0,1] => [4]
    return A.nonzero()[0]
def sp_matrix(m,n = 1, dtype = 'float'):
    # returns a sparse matrix with m rows and n columns, with the dtype
    # We use dok_matrix for sparse matrixies
    return sp.lil_matrix((m,n),dtype=dtype)
def sp_dot_array(sp_m, A):
    #Efficient dot product of matrix sp_m in shape of p-by-1 and array A with p elements
    assert sp_m.shape[0] == len(A)
    ind = sp_m.nonzero()[0]
    if len(ind) == 0:
        return 0
    if sp_m.dtype == bool:
        #Just sum the corresponding indexes of theta
        return sum(A[ind])
    else:
        # Multiply by feature values since they are not binary
        return sum([A[i]*sp_m[i,0] for i in ind])
def sp_dot_sp(sp_1, sp_2):
    #Efficient dot product of matrix sp_m in shape of p-by-1 and array A with p elements
    assert sp_1.shape[0] == sp_2.shape[0] and sp_1.shape[1] == 1 and sp_2.shape[1] == 1 
    ind_1 = sp_1.nonzero()[0]
    ind_2 = sp_2.nonzero()[0]
    if len(ind_1)*len(ind_2) == 0:
        return 0
    
    ind = intersect1d(ind_1,ind_2)
    # See if they are boolean
    if sp_1.dtype == bool and sp_2.dtype == bool:
        return len(ind)
    sp_bool = None
    if sp_1.dtype == bool:
        sp_bool = sp_1
        sp      = sp_2
    if sp_2.dtype == bool:
        sp_bool = sp_2
        sp      = sp_1
    if sp_bool is None:
        # Multiply by feature values since they are not binary
        return sum([sp_1[i,0]*sp_2[i,0] for i in ind])
    else:
        return sum([sp[i,0] for i in ind])
def sp_add2_array(sp,A):
    # sp is a sparse matrix p-by-1
    # A is an array of len p
    # this function return an array corresponding to A+sp
    ind = sp.nonzero()[0]
    for i in ind:
        A[i] += sp[i,0]
    return A
def checkNCreateDirectory(fullfilename):
    # See if a fullfilename exists if not create the required directory
    path,char,filename = fullfilename.rpartition('/')
    if not os.path.exists(path):
        os.makedirs(path)
def hasFunction(object,methodname):
    method = getattr(object, methodname, None)
    return callable(method)
class Logger(object):
    buffer = ''         # You can print into a logger without initializing its filename. Whenever the filename is set, the buffer is flushed to the output.
    def save(self,filename):
            checkNCreateDirectory(filename)
            self.file = open(filename,'w')
            lastprint = 'Log\t=> %s' % filename
            print lastprint
            self.buffer += lastprint
            self.file.write(self.buffer)
            buffer = ''
            self.file.close()
    def log(self,str):
    # Print something both in output and in a file
        print str
        self.buffer += str +'\n'
    def line(self):
        self.log(SEP_LINE)
class MergedData(object):
    AXES = ['Learning Steps','Return','Time(s)','Features','Steps','Terminal']
    def __init__(self,path, output_path = None, colors = ['r','b','g','k'],bars=1):
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        self.means                  = []
        self.std_errs               = [] 
        self.bars                   = bars  #Draw bars?
        self.colors                 = colors
        self.path                   = path
        self.output_path            = path if output_path == None else output_path
        self.exp_paths              = os.listdir(path)
        self.exp_paths              = [p for p in self.exp_paths if os.path.isdir(path+'/'+p) and os.path.exists(path+'/'+p+'/1-out.txt')]
        self.exp_num                = len(self.exp_paths) 
        self.means                  = []
        self.std_errs               = [] 
        self.fig                    = pl.figure(1)
        self.datapoints_per_graph   = None # Number of datapoints to be shown for each graph (often this value is 10 corresponding to 10 performance checks)
        for exp in self.exp_paths:
            means, std_errs = self.parseExperiment(exp)
            self.means.append(means)
            self.std_errs.append(std_errs)
    def parseExperiment(self,exp):
        # Parses all the files in form of <number>-results.txt and return 
        # two matrices corresponding to mean and std_err
        path        = self.path + '/'+exp
        files       = glob.glob('%s/*-results.txt'%path)
        samples_num = len(files)
        if samples_num == 0:
            print 'Error: %s is empty!' % path
        print "%s => %d Samples" % (exp,samples_num)
        #read the first file to initialize the matricies
        rows, cols  = loadtxt(files[0]).shape
        samples     = zeros((rows,cols,samples_num)) 
        for i,f in enumerate(files):
            samples[:,:,i] = loadtxt(files[i])
        _,self.datapoints_per_graph,_ = samples.shape
        return mean(samples,axis=2),std(samples,axis=2)/sqrt(samples_num)
    def plot(self,Y_axis,X_axis = 'Learning Steps'):
        self.fig.clear()
        min_ = +inf
        max_ = -inf    
        if Y_axis in self.AXES:
            y_ind = self.AXES.index(Y_axis)
        else:
            print 'unknown Y_axis = %s', Y_axis
        if X_axis in [self.AXES[0],self.AXES[2]]:
            x_ind = self.AXES.index(X_axis)
        else:
            print 'unknown X_axis = %s', X_axis
        Xs      = zeros((self.exp_num,self.datapoints_per_graph))
        Ys      = zeros((self.exp_num,self.datapoints_per_graph))
        Errs    = zeros((self.exp_num,self.datapoints_per_graph))
        for i in range(self.exp_num):
            X   = self.means[i][x_ind,:]
            Y   = self.means[i][y_ind,:]
            Err = self.std_errs[i][y_ind,:]
            pl.plot(X,Y,'-o', linewidth = 2,alpha=.5,color = self.colors[i],)
            if self.bars:
                pl.fill_between(X, Y-Err, Y+Err,alpha=.1, color = self.colors[i])
                max_ = max(max(Y+Err),max_); min_ = min(min(Y-Err),min_)
            else:
                max_ = max(Y.max(),max_); min_ = min(Y.min(),min_)
            Xs[i,:]     = X
            Ys[i,:]     = Y
            Errs[i,:]   = Err
        pl.xlim(0,max(Xs[:,-1]))
        pl.ylim(min_-.1*abs(max_-min_),max_+.1*abs(max_-min_))
        pl.xlabel(X_axis,fontsize=16)
        pl.ylabel(Y_axis,fontsize=16)
        pl.show()
        self.save(Y_axis,X_axis,Xs,Ys,Errs)
    def save(self,Y_axis,X_axis,Xs,Ys,Errs):
        fullfilename = self.output_path + '/' +Y_axis+'-by-'+X_axis
        checkNCreateDirectory(fullfilename)
        self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1)
        finalArray = vstack((Xs,Ys,Errs))
        savetxt(fullfilename+'.txt',finalArray, fmt='%0.4f', delimiter='\t')
        print "==================\nSaved Outputs at\n1. %s\n2. %s" % (fullfilename+'.txt',fullfilename+'.pdf')
            

createColorMaps()
FONTSIZE = 15
SEP_LINE = "="*60
rc('font',**{'family':'serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
mpl.rcParams['font.size'] = 15.
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 15.
mpl.rcParams['xtick.labelsize'] = 15.
mpl.rcParams['ytick.labelsize'] = 15.
os.environ['PATH'] += ':/usr/texbin'
os.environ['PATH'] += ':/usr/share/texmf'