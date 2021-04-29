#!/usr/bin/python
# example of python sript that calculate the trajectory of a free particle
# the free particle is initially at (0,0), with a velocity vector \bold{v_0}
# making an angle of 60 degrees with the x-axis
# Is a running code necessarily giving you the correct answer? If not, where is the mistake?

import sys
import math
import matplotlib.pyplot as plt

class Particle:

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

# simulation conditions -> could be defined as a function in a different module (settings.py -> import.settings and then import.set())
m = 1          # number of particles
imax = 200     # number of iterations
dt = 0.001     # time step


# initial condition -> Should be defined as a function in a difference module (initialize.py -> import initialize)
x = 0.         # x_0 [a.u.]
y = 0.         # y_0 [a.u.]
v0 = 1.        # v_0 [a.u.]
initangle = 60 # initial angle
theta = initangle*math.asin(1.)/180.
vx =  v0 * math.cos(theta)
vy = v0 * math.sin(theta)

part = Particle(x, y, vx, vy)

fx = 0
fy = -m * 9.8

#open output file
file = open("output.txt", "w")
p = []
p.append(Particle(x,y,vx,vy))
# time integration
itime = 0
while itime < imax:
    x = part.x                              # x(t)
    y = part.y                              # y(t)
    vx = part.vx                            # vx(t)
    vy = part.vy                            # vy(t)
    x1 = x + vx * dt + (force[0]/2./m) * dt * dt  # x(t+dt) = x(t) + vx(t) * dt + (fx(t)/2m) * dt*dt
    y1 = y + vy * dt + (force[1]/2./m) * dt * dt  # y(t+dt) = y(t) + vy(t) * dt + (fy(t)/2m) * dt*dt
    vx1 = vx + dt/2./m * (fx + fx)          # vx(t+dt) = vx(t) + dt/2m * (fx(t) + fy(t+dt))
    vy1 = vy + dt/2./m * (fy + fy)          # vy(t+dt) = vy(t) + dt/2m * (fy(t) + fy(t+dt))


    part = Particle(x1, y1, vx1, vy1)       # update particle's position and velocity
    p.append(Particle(x,y,vx,vy))           # save trajectory and velocities in p
    file.write(str(itime) + " " + str(x1) + " " + str(y1) + "\n") # write trajectory and velocity in output file
    itime += 1

file.close()
xx = []
yy = []
for i in range(len(p)):
    xx.append(p[i].x)
    yy.append(p[i].y)
plt.plot(xx,yy)
plt.xlabel('x [a.u.]')
plt.ylabel('y [a.u.]')
plt.show()

del p
del xx
del yy
