#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* SYSTEM MODULE *
Contains the system class

Latest update: May 8th 2021
"""

import numpy as np
import settings
import const
import force
import integrator
import itertools
from numba import jit, njit, vectorize

# Particles variables (position, velocity, net force and mass)

dim = None #dimension of the system (2D or 3D)
alat = None #Lattice parameter
rho = None #Number density
p = None

mass = None
N = None
L = None #Box dimensions (per edge)
pos = None
force = None

# Energy and Thermodynamics variables
energy = None
kinetic = None
potential = None
T = None
cT = None


# Routine for generating random velocities
# //default is uniform distribution
# //boltzmann distribution needs to be implemented
def vel_random(default = 'uniform'):
    global vel

    if(default == 'uniform'):
        vel = np.random.uniform(-1.0, 1.0, (N,dim))
    elif(default == 'boltzmann'):
        print("yet to be implemented")

# Routine for shifting the velocities. This is done in order to cancel
# the linear momentum of the system.
# //The mean over each axis is computed and subtracted from the velocity values
def vel_shift():
    global vel

    mean = np.sum(vel, axis = 0)/N
    vel[:,0] -= mean[0]
    vel[:,1] -= mean[1]
    if (dim == 3):
        vel[:,2] -= mean[2]

# Routine for placing particles on a cubic lattice of given parameter a_lattice.
# Let us assume for simplicity that the relationship between a_lattice and L (box dim)
# is given by L
# //the default a_lattice is defined globally
def distribute_position_cubic_lattice():
    global L, pos
    if(L == None):
        if(rho == None):
            print("Error: unspecified number density and box volume. Unable to distribute positions over cubic lattice")
            sys.exit()
        else:
            L = np.power(N/rho, 1/dim)

    # generates position over a cubic lattice of a_lat = 1 (only positive positions)
    S_range = list(range(0,int(np.power(N, 1/dim))))
    cubic_lattice = np.array(list(itertools.product(S_range, repeat=dim)))

    # rescaling and shifting
    pos = cubic_lattice * (L/np.cbrt(N)) + (L/(2*np.cbrt(N)))

# Routine for rescaling the velocities in order to achieve
# the target temperature for the system using Eq (7.21)
# //note: re-computing the temperature might not always be necessary
def vel_rescale(temp):
    global vel
    v_ave = np.sum(np.multiply(vel, vel))/N
    vel *= np.sqrt(3*temp*const.KB/v_ave)

def compute_energy():
    system.energy = system.kinetic + system.potential
