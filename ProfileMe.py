from main import *
import cProfile
print 'Profiling'
cProfile.run('main()', 'Profiling/main.profile')