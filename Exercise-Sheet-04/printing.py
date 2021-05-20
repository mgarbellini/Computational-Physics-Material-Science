#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* PRINTING MODULE *
Contains all the routines for printing and saving computational
data to file. This includes basic plotting and ovito style outputs,
as well as printing routines for debugging

Latest update: May 14th 2021
"""

import sys
import time
import settings
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # PLOTTING  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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

def plot_energy(kind, run):
    type = ["Kinetic", "Potential", "Total"]
    typeof = ["equil", "prod"]
    type_small = ["K", "P", "E"]
    filename = "./LJMD_" + str(system.N) + "_" + typeof[run] +"_energy.txt"
    E = np.loadtxt(filename, usecols = kind)
    x = np.arange(len(E))*20
    mean_E = np.mean(E)

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,E, label= type[kind] + " Energy")
        ax.hlines(mean_E, 0, len(x)*20 , color ="red", label = "Mean Energy")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel(type_small[kind] + ' [reduced units]')

        #ax.set_ylim([155.4, 155.6])
        ax.legend()
        ax.set_title(type[kind] + " energy during " + typeof[run])
        fig.savefig('./figures/'+ str(system.N)+'_'+ typeof[run] + '_' + type[kind] + '_Energy.pdf')


def plot_velocity(run):
    xcols = []
    ycols = []
    zcols = []
    typeof = ["equil", "prod"]

    iter = 0
    while iter<system.N*3:
        xcols.append(iter)
        ycols.append(iter+1)
        zcols.append(iter+2)
        iter+=3

    velx = np.loadtxt("./LJMD_" + str(system.N) + "_"+ typeof[run]+ "_velocity.txt", usecols = xcols)
    vely = np.loadtxt("./LJMD_" + str(system.N) + "_"+ typeof[run]+ "_velocity.txt", usecols = ycols)
    velz = np.loadtxt("./LJMD_" + str(system.N) + "_"+ typeof[run]+ "_velocity.txt", usecols = zcols)

    sum_x = np.sum(velx*velx, axis = 1)
    sum_y = np.sum(vely*vely, axis = 1)
    sum_z = np.sum(velz*velz, axis = 1)

    x = np.arange(len(sum_x))*20

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,sum_x, label="$\sum_{i}^N v_{i,x}^2$")
        ax.plot(x,sum_y, label="$\sum_{i}^N v_{i,y}^2$")
        ax.plot(x,sum_z, label="$\sum_{i}^N v_{i,z}^2$")

        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('$\sum v^2 $ [reduced units]')
        ax.legend()
        ax.set_title("Sum of squared velocities")
        fig.savefig('./figures/' + str(system.N)+ '_'+ typeof[run]+'_velocities.pdf')

def plot_rdf(histogram, bins):
    #plot radial distribution function
    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(bins,histogram, label="$g(r)$")
        #ax.autoscale(tight=True)
        ax.set_xlabel('r $[\sigma]$')
        ax.set_ylabel('$g(r)$')
        ax.set_xlim([0,0.5*system.L])
        ax.legend()
        ax.set_title("Radial Distribution Function")
        fig.savefig('./figures/rdf.pdf')

def plot_coord(histogram, bins):
    #plot radial distribution function
    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(bins,histogram, label="$n_c(r)$")
        #ax.autoscale(tight=True)
        ax.set_xlabel('r $[\sigma]$')
        ax.set_ylabel('$n_c(r)$')
        ax.set_xlim([0,np.max(bins)])
        ax.legend()
        ax.set_title("Running coordination number")
        fig.savefig('./figures/coord.pdf')




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # PRINTING TO FILE  # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def print_system(type):
    global eq_energy, eq_temp, eq_pos, eq_vel
    global prod_energy, prod_temp, prod_pos, prod_vel

    if(type=='equilibration'):

        #print energies
        eq_energy.write(str(np.around(system.kinetic,digits)) + " " + str(np.around(system.potential,digits)) + " " + str(np.around(system.kinetic+system.potential,digits)) + "\n")

        #print positions
        for i in range(system.N):
            eq_pos.write(str(np.around(system.pos[i,0],digits)) + " " + str(np.around(system.pos[i,1],digits)) + " " + str(np.around(system.pos[i,2],digits)) + " ")
        eq_pos.write("\n")

        #print velocities
        for i in range(system.N):
            eq_vel.write(str(np.around(system.vel[i,0],digits)) + " " + str(np.around(system.vel[i,1],digits)) + " " + str(np.around(system.vel[i,2],digits)) + " ")
        eq_vel.write("\n")

    elif (type=='production'):

        #print energies
        prod_energy.write(str(np.around(system.kinetic,digits)) + " " + str(np.around(system.potential,digits)) + " " + str(np.around(system.kinetic+system.potential,digits)) + "\n")

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

def print_ovito(filename):
    ovito = open(filename, 'w')
    ovito.write("ITEM: TIMESTEP \n")
    ovito.write("%i \n" % 0)
    ovito.write("ITEM: NUMBER OF ATOMS \n")
    ovito.write( "%i \n" % system.N)
    ovito.write("ITEM: BOX BOUNDS pp pp pp \n")
    ovito.write("%e %e \n" % (system.L, system.L))
    ovito.write("%e %e \n" % (system.L, system.L))
    ovito.write("%e %e \n" % (system.L, system.L))
    ovito.write("ITEM: ATOMS id x y z \n")
    for i in range(0, system.N):
        ovito.write("%i %e %e %e \n" % (i, system.pos[i,0],system.pos[i,1], 0))

    ovito.close()
