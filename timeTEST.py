from Tools import *
from main import *
from Representations import *
from Domains import *
import timeit

t = timeit.Timer(stmt="main(1,0)",setup="from __main__ import *")
print t.timeit(number=1)  
