#!/usr/bin/env python3
"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 01 - Planetary Evolution

The following is the code for plotting the computational results.
List of plots:
(1) Planets position in 50 years (3D) (VV)
(2) Planets position in 100 years (3D) (VV)
(3) Planets position in 50 years (xy projection) (VV)
(4) Planets position in 100 years (xy projection) (VV)
(5) Pluto trajectory first 100 years (Velocity-Verlet)
(6) Pluto trajectory second 100 years (Velocity-Verlet)
(7) Pluto trajectory over 300 years (Velocity-Verlet)
(8) Pluto trajectory over 300 years (Verlet)
(9) Mars trajectory over 2 years (Verlet)
(10) Total energy of system with Euler
(11) Pluto trajectory over 300 years (Euler)


Indices for the loaded data
0,1,2    !** Sun
3,4,5    !** Mercury
6,7,8    !** Venus
9,10,11  !** Earth
12,13,14 !** Mars
15,16,17 !** Jupiter
18,19,20 !** Saturn
21,22,23 !** Uranus
24,25,26 !** Nepture
27,28,29 !** Pluto
30,31,32 !** energy, kinetic, potential
"""
import numpy as np
import sys
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Computational results import
euler = np.loadtxt("euler_300y.txt")
verlet = np.loadtxt("verlet_300y.txt")
velocity = np.loadtxt("velocity_verlet_300y.txt")
name = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']


# # # # # # #
#  PLOT (1) #
# # # # # # #


iter_50 = 50*2*365
fig1 = plt.figure(figsize=(12,9))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
p=0
while p<30:
    ax1.scatter(data_euler[iter_50,p], data_euler[iter_50,p+1],data_euler[iter_50,p+2], c="C0", s=20)
    p+=3

ax1.set_xlabel('x [a.u.]')
ax1.set_ylabel('y [a.u.]')
ax1.set_zlabel('z [a.u.]')
ax1.set_title("Planets positions after 50 years")
ax1.legend()
fig1.savefig('./planet_position_50y_3d.pdf')


# # # # # # #
#  PLOT (2) #
# # # # # # #

iter_100 = 100*2*365
fig2 = plt.figure(figsize=(12,9))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
p=0
while p<30:
    ax2.scatter(data_euler[iter_100,p], data_euler[iter_100,p+1],data_euler[iter_100,p+2], c="C0", s=20)
    p+=3

ax2.set_xlabel('x [a.u.]')
ax2.set_ylabel('y [a.u.]')
ax2.set_zlabel('z [a.u.]')
ax2.set_title("Planets positions after 50 years")
ax2.legend()
fig2.savefig('./planet_position_100y_3d.pdf')


# # # # # # #
#  PLOT (3) #
# # # # # # #

with plt.style.context(['science']):
    iter_50 = 50*2*365
    fig3 = plt.figure(figsize=(12,9))
    ax3 = fig3.add_subplot()
    ax3.scatter(0,0, c="red", label="Sun") #sun position
    p=3
    n = 1
    while p<30:
        ax3.scatter(velocity[iter_50,p], velocity[iter_50,p+1], s=20, label = name[n] )
        p+=3
        n+=1

    ax3.set_xlabel('x [a.u.]')
    ax3.set_ylabel('y [a.u.]')
    ax3.set_title("Planets positions after 50 years (x-y projection)")
    ax3.legend()
    fig3.savefig('./planet_position_50y.pdf')


# # # # # # #
#  PLOT (4) #
# # # # # # #

with plt.style.context(['science']):
    iter_100 = 100*2*365
    fig4 = plt.figure(figsize=(12,9))
    ax4 = fig4.add_subplot()
    ax4.scatter(0,0, c="red", label="Sun") #sun position
    p=3
    n = 1
    while p<30:
        ax3.scatter(velocity[iter_100,p], velocity[iter_100,p+1], s=20, label = name[n] )
        p+=3
        n+=1

    ax4.set_xlabel('x [a.u.]')
    ax4.set_ylabel('y [a.u.]')
    ax4.set_title("Planets positions after 100 years (x-y projection)")
    ax4.legend()
    fig4.savefig('./planet_position_100y.pdf')


# # # # # # #
#  PLOT (5) #
# # # # # # #
iter100 = 100*2*365
fig5 = plt.figure(figsize=(12,9))
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
ax5.scatter(data_euler[0:iter100,27], data_euler[0:iter100,28],data_euler[0:iter100,29], c="C0", s=20, label ='Pluto')
ax5.set_xlabel('x [a.u.]')
ax5.set_ylabel('y [a.u.]')
ax5.set_zlabel('z [a.u.]')
ax5.set_title("Pluto trajectory [0,100] years with Velocity-Verlet integrator")
ax5.legend()
fig5.savefig('./pluto_trajectory_100y.pdf')


# # # # # # #
#  PLOT (6) #
# # # # # # #
iter200 = 200*2*365
fig6 = plt.figure(figsize=(12,9))
ax6 = fig6.add_subplot(111, projection='3d')
ax6.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
ax6.scatter(data_euler[iter100:iter200,27], data_euler[iter100:iter200,28],data_euler[iter100:iter200,29], c="C0", s=20, label = 'Pluto')
ax6.set_xlabel('x [a.u.]')
ax6.set_ylabel('y [a.u.]')
ax6.set_zlabel('z [a.u.]')
ax6.set_title("Pluto trajectory [101,200] years with Velocity-Verlet integrator")
ax6.legend()
fig6.savefig('./pluto_trajectory_200y.pdf')


# # # # # # #
#  PLOT (7) #
# # # # # # #
fig7 = plt.figure(figsize=(12,9))
ax7 = fig7.add_subplot(111, projection='3d')
ax7.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
ax7.scatter(data_euler[:,27], data_euler[:,28],data_euler[:,29], c="C0", s=20, label = 'Pluto')
ax7.set_xlabel('x [a.u.]')
ax7.set_ylabel('y [a.u.]')
ax7.set_zlabel('z [a.u.]')
ax7.set_title("Pluto trajectory over 300 years  (Velocity-Verlet integrator)")
ax7.legend()
fig7.savefig('./pluto_trajectory_velocity_verlet_300y.pdf')


# # # # # # #
#  PLOT (8) #
# # # # # # #
fig8 = plt.figure(figsize=(12,9))
ax8 = fig8.add_subplot(111, projection='3d')
ax8.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
ax8.scatter(data_euler[:,27], data_euler[:,28],data_euler[:,29], c="C0", s=20, label = 'Pluto')
ax8.set_xlabel('x [a.u.]')
ax8.set_ylabel('y [a.u.]')
ax8.set_zlabel('z [a.u.]')
ax8.set_title("Pluto trajectory over 300 years (Verlet Integrator)")
ax8.legend()
fig8.savefig('./pluto_trajectory_verlet_300y.pdf')


# # # # # # #
#  PLOT (9) #
# # # # # # #
iter2 = 2*365*2
fig9 = plt.figure(figsize=(12,9))
ax9 = fig9.add_subplot(111, projection='3d')
ax9.scatter(0,0,0, c="red", s=50, label="Sun") #sun position
ax9.scatter(data_euler[0:iter2,12], data_euler[0:iter2,13],data_euler[0:iter2,14], c="C0", s=20, label ='Mars')
ax9.set_xlabel('x [a.u.]')
ax9.set_ylabel('y [a.u.]')
ax9.set_zlabel('z [a.u.]')
ax9.set_title("Mars trajectory over 2 years (Verlet Integrator)")
ax9.legend()
fig9.savefig('./mars_trajectory_2y.pdf')


# # # # # # #
#  PLOT (10) #
# # # # # # #
items = np.arange(start=0, stop=len(euler[:,30]), step=1)
with plt.style.context(['science']):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(items,euler[:,30], label = "Total Energy")

    ax.legend(title='Energy conservation over 300 years (Euler integrator)')
    #ax.set_xlim([-10E11,10E11])
    #ax.set_ylim([-10E11,10E11])
    #ax.autoscale(tight=True)
    fig.savefig('./euler_energy.pdf')
