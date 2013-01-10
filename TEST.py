from sklearn import svm
from Tools import *
from Domains import PitMaze
    
class foo():
    def __init__(self,x):
        self.x = x
q = PriorityQueueWithNovelty()
q.push(1,foo(1))
q.push(2,foo(2))
q.push(1,foo(3))
q.push(10,foo(4))

q.show()
sorted = q.toList()
for x in sorted:
    print x.x

