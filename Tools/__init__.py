from GeneralTools import *
from joblib import Memory
from Logger import Logger
from Merger import Merger
from PriorityQueueWithNovelty import PriorityQueueWithNovelty
from GeneralTools import __rlpy_location__
import os
memory = Memory(os.path.join(__rlpy_location__,"cache"))
