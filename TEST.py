from Tools import *
from Representations import *
from Domains import *

#for i, (stylename, styleclass) in enumerate(sorted(styles.items())):
#    x = 3.2 + (i//nrow)*4
#    y = (figheight - 0.7 - i%nrow) # /figheight
#    p = mpatches.Circle((x, y), 0.2, fc="w")
#    ax.add_patch(p)
#


states  = 5
Y       = 1
fig1 = pl.figure(1, (states*2, 2))

ax = fig1.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)

ax.set_xlim(0, states*2)
ax.set_ylim(0, 2)

ax.add_patch(mpatches.Circle((1+2*(states-1), Y), .55, fc="w")) #Make it double circle
RADIUS = .5
states_fig = [mpatches.Circle((1+2*i, Y), RADIUS, fc="w") for i in range(states)]
for i in range(states):
    ax.add_patch(states_fig[i])
Shift = .3
for i in range(states-1):
    fromAtoB(1+2*i+Shift,Y+Shift,1+2*(i+1)-Shift, Y+Shift)
    if i != states-2: fromAtoB(1+2*(i+1)-Shift,Y-Shift,1+2*i+Shift, Y-Shift, 'r')
#fromAtoB(.75,Y-2*Shift,.75,Y+1.5*Shift,'r',connectionstyle='arc,angleA=-90,angleB=0,armA=0,armB=40,rad=0')
fromAtoB(.75,Y-1.5*Shift,.75,Y+1.5*Shift,'r',connectionstyle='arc3,rad=-1.2')
states_fig[states-2].set_facecolor('b')



pl.draw()
pl.show()
