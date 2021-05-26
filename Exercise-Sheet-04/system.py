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
n = None #Number of particles for given axis n.shape = (3,)
N = None
L = None #Box dimensions (per edge)
pos = None
force = None
f_wall_dw = None #force on lower wall
f_wall_up = None #force on upper wall

# Energy and Thermodynamics variables
energy = None
kinetic = None
potential = None
T = None
cT = None

external_force = False

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
"""
def distribute_position_cubic_lattice():
    global pos

    # generates position over a cubic lattice of a_lat = 1 (only positive positions)
    S_range = list(range(0,int(np.power(N, 1/dim))))
    cubic_lattice = np.array(list(itertools.product(S_range, repeat=dim)))

    # rescaling and shifting
    pos = cubic_lattice * (L/np.cbrt(N)) + (L/(2*np.cbrt(N)))
"""

def lattice_position():
    global pos

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    places_z = np.linspace(0, L[2], num=n[2], endpoint = False)
    places_z += places_z[1]*0.5
    n_part = 0

    for i,j,k in itertools.product(list(np.arange(n[0])), list(np.arange(n[1])), list(np.arange(n[2]))):

        x[n_part] = alat*i
        y[n_part] = alat*j
        z[n_part] = places_z[k]
        n_part += 1

    pos[:,0] = x
    pos[:,1] = y
    pos[:,2] = z


# Routine for rescaling the velocities in order to achieve
# the target temperature for the system using Eq (7.21)
# //note: re-computing the temperature might not always be necessary
def vel_rescale(temp):
    global vel
    v_ave = np.sum(np.multiply(vel, vel))/N
    vel *= np.sqrt(3*temp*const.KB/v_ave)

def compute_energy():
    system.energy = system.kinetic + system.potential
