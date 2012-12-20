from Tools import *
from Representations import *
from Domains import *

import numpy as np

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#ax.contourf3D(X, Y, Z, rstride=10, cstride=10)
ax.plot_surface(X, Y, Z)
pl.ioff()
pl.show()
