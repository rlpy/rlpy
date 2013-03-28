from Tools import *
from Domains import *
from Representations import *


def hasResults(path):
    return len(glob.glob(os.path.join(path, '*-results.txt'))) >= 1

path = '.'
for p in os.walk(path):
    dirname = p[0]
    if not '/.' in dirname and hasResults(dirname):
        print dirname