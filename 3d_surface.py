from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import time


seed(0)
npts = 200
ngridx = 100
ngridy = 100
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)

# griddata and contour.
start = time.clock()
plt.subplot(211)
xi = np.linspace(-2.1,2.1,ngridx)
yi = np.linspace(-2.1,2.1,ngridy)
zi = griddata(x,y,z,xi,yi,interp='linear')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#X, Y, Z = axes3d.get_test_data(0.05)


#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#ax.contourf(X, Y, Z,1000)
ax.contourf(xi, yi, zi,1000)

plt.show()

