#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from heapq import heappush, heappop
from Tools import deepcopy


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
