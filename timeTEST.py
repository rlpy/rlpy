from Tools import *
from main import *
from Representations import *
from Domains import *
import timeit

def f1():
    x = random.random_integers(0,L-1,L)
    y = [L]*L
    return vec2id(x,y)
def f2():
    x = random.random_integers(0,L-1,L)
    y = [L]*L
    return vec2id2(x,y)

random.seed(999)
L = 10
t = timeit.Timer(stmt="f1()",setup="from __main__ import *")
print t.timeit(number=100)  
t = timeit.Timer(stmt="f2()",setup="from __main__ import *")
print t.timeit(number=100)  

