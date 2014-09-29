"""Priority Queue with Novelty"""

from heapq import heappush, heappop
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class PriorityQueueWithNovelty(object):

    """This is a priority queue where it is sorted based on priority and then then novelty of elements
    First Order: The Lower the priority the better
    Second Order: The newer the item the better
    Example:
    >>> H = PriorityQueueWithNovelty()
    >>> H.put(1,"Q1")
    >>> H.put(2,"O2")
    >>> H.put(1,"O3")
    >>> H.put (10,"O4")
    >>> H.toList()
    ["O3", "O1", "O2", "O4"]

    Adopted from http://stackoverflow.com/questions/9289614/how-to-put-items-into-priority-queues
    """

    def __init__(self):
        self._h = []
        self.counter = 0
        self.cache = None

    def push(self, priority, item):
        heappush(self._h, (priority, self.counter, item))
        self.counter -= 1
        self.cache = None

    def pop(self):
        _, _, item = heappop(self._h)
        self.cache = None
        return item

    def empty(self):
        return len(self._h) == 0

    def toList(self):
        if self.cache is None:
            temp = list(self._h)
            self.cache = [heappop(temp)[2] for i in range(len(temp))]
        return self.cache

    def show(self):
        temp = list(self._h)
        for i in range(len(temp)):
            p, c, x = heappop(temp)
            print "Priority = %d, Novelty = %d, Obj = %s" % (p, c, str(x))

    def __deepcopy__(self, memo):
        new_q = PriorityQueueWithNovelty()
        new_q._h = deepcopy(self._h)
        new_q.counter = self.counter
        return new_q
