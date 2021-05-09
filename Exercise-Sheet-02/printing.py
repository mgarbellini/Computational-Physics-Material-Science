#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* PRINTING MODULE *
Contains all the routines for printing and saving computational
data to file. This includes basic plotting and ovito style outputs,
as well as printing routines for debugging

Latest update: May 8th 2021
"""

import sys
import time
import numpy as np
import system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Filenames for output
eq_print = None
eq_energy_file = None
eq_temp_file = None
eq_pos_file = None
eq_vel_file = None

prod_print = None
prod_energy_file = None
prod_temp_file = None
prod_pos_file = None
prod_vel_file = None

# files variable
eq_energy = None
eq_temp = None
eq_pos = None
eq_vel = None
prod_energy = None
prod_temp = None
prod_pos = None
prod_vel = None

# output numbers precision
digits = 4
ovito = None
ovito_file = "LJ_simulation.txt"

# Routine for plotting the inital lattice positions
# //works for any position
def plot_system(filename, iter):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(system.pos[:,0], system.pos[:,1], system.pos[:,2], s = 40)
    ax.set_xlim([0,system.L])
    ax.set_ylim([0,system.L])
    ax.set_zlim([0,system.L])
    fig.savefig('./pdf/' + filename + '_' + str(iter) + '.pdf')

def print_system(type):
    global eq_energy, eq_temp, eq_pos, eq_vel
    global prod_energy, prod_temp, prod_pos, prod_vel

    if(type=='equilibration'):
        # print temperature
        eq_temp.write(str(np.around(system.cT,5)) + "\n")

        #print energies
        eq_energy.write(str(np.around(system.kinetic,digits)) + " " + str(np.around(system.potential,digits)) + " " + str(np.around(system.energy,digits)) + "\n")

        #print positions
        for i in range(system.N):
            eq_pos.write(str(np.around(system.pos[i,0],digits)) + " " + str(np.around(system.pos[i,1],digits)) + " " + str(np.around(system.pos[i,2],digits)) + " ")
        eq_pos.write("\n")

        #print velocities
        for i in range(system.N):
            eq_vel.write(str(np.around(system.vel[i,0],digits)) + " " + str(np.around(system.vel[i,1],digits)) + " " + str(np.around(system.vel[i,2],digits)) + " ")
        eq_vel.write("\n")

    elif (type=='production'):
        # print temperature
        prod_temp.write(str(np.around(system.cT,5)) + "\n")

        #print energies
        prod_energy.write(str(np.around(system.kinetic,digits)) + " " + str(np.around(system.potential,digits)) + " " + str(np.around(system.energy,digits)) + "\n")

        #print positions
        for i in range(system.N):
            prod_pos.write(str(np.around(system.pos[i,0],digits)) + " " + str(np.around(system.pos[i,1],digits)) + " " + str(np.around(system.pos[i,2],digits)) + " ")
        prod_pos.write("\n")

        #print velocities
        for i in range(system.N):
            prod_vel.write(str(np.around(system.vel[i,0],digits)) + " " + str(np.around(system.vel[i,1],digits)) + " " + str(np.around(system.vel[i,2],digits)) + " ")
        prod_vel.write("\n")
    else:
        print("Error: undefined/unspecified printing scheme")

def openfiles(type):

    if(type=="equilibration"):
        global eq_energy, eq_temp, eq_pos, eq_vel
        eq_energy = open(eq_energy_file, 'w')
        eq_temp = open(eq_temp_file, 'w')
        eq_pos = open(eq_pos_file, 'w')
        eq_vel = open(eq_vel_file, 'w')

        eq_energy.write("# kinetic - potential - total energy \n")
        eq_temp.write("#temperature \n")
        eq_pos.write("#x1, y1, z1, ...., xN, yN, zN \n")
        eq_vel.write("#vx1, vy1, vz1, ...., vxN, vyN, vzN \n")

    elif(type=="production"):
        global prod_energy, prod_temp, prod_pos, prod_vel
        prod_energy = open(prod_energy_file, 'w')
        prod_temp = open(prod_temp_file, 'w')
        prod_pos = open(prod_pos_file, 'w')
        prod_vel = open(prod_vel_file, 'w')

        prod_energy.write("# kinetic - potential - total energy \n")
        prod_temp.write("#temperature \n")
        prod_pos.write("#x1, y1, z1, ...., xN, yN, zN \n")
        prod_vel.write("#vx1, vy1, vz1, ...., vxN, vyN, vzN \n")

    else:
        print("Error: undefined/unspecified printing scheme")

def closefiles(type):
    if(type=="equilibration"):
        global eq_energy, eq_temp, eq_pos, eq_vel
        eq_energy.close()
        eq_temp.close()
        eq_pos.close()
        eq_vel.close()

    elif(type=="production"):
        global prod_energy, prod_temp, prod_pos, prod_vel
        prod_energy.close()
        prod_temp.close()
        prod_pos.close()
        prod_vel.close()

    else:
        print("Error: undefined/unspecified printing scheme")

def open_ovito():
    global ovito
    ovito = open(ovito_file, 'w')
def close_ovito():
    global ovito
    ovito.close()

def print_ovito(iter):
    global ovito
    print("yet to be implemented")
