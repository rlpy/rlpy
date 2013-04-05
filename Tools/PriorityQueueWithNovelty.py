from GeneralTools import *
class PriorityQueueWithNovelty():
    # This is a priority queue where it is sorted based on priority and then then novelty of elements
    # First Order: The Lower the priority the better
    # Second Order: The newer the item the better
    # Example:
    # put (1,<O1>)
    # put (2,<O2>)
    # put (1,<O3>)
    # put (10,<O4>)
    # => multiple get() : O3,O1,O2,O4
    # Adopted from http://stackoverflow.com/questions/9289614/how-to-put-items-into-priority-queues
    def __init__(self):
        self.h = []
        self.counter = 0
    def push(self, priority, item):
        heappush(self.h,(priority, self.counter, item))
        self.counter -= 1
    def pop(self):
        _, _, item = heappop(self.h)
        return item
    def empty(self):
        return len(self.h) == 0
    def toList(self):
        temp = list(self.h)
        return [heappop(temp)[2] for i in range(len(temp))]
    def show(self):
        temp = list(self.h)
        for i in range(len(temp)):
            p,c,x = heappop(temp)
            print "Priotiry = %d, Novelty = %d, Obj = %s" % (p,c,str(x))
    def __deepcopy__(self,memo):
        new_q           = PriorityQueueWithNovelty()
        new_q.h         = deepcopy(self.h)
        new_q.counter   = self.counter
        return new_q