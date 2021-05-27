#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* SETTINGS MODULE *
Contains all the settings for a given simulation.
At the first call of settings.init() all specified variables
are initialized and available.

Latest update: May 8th 2021
"""

import system
import force
import printing
import numpy as np
from numba import jit, njit, vectorize
import numba

DT = None
iter_equ = None
iter_prod = None
rescaling_freq = None



# Routine for initializing all variables of the System
def init():

    # LJ POTENTIAL VARIABLES
    force.epsilon = 1.
    force.sigma = 1.
    force.cutoff = 4 #in units of sigma
    force.epsilon_wall = force.epsilon
    force.sigma_wall = force.sigma/5
    force.cutoff_wall = 2.5*force.sigma_wall

    # SYSTEM VARIABLES
    system.n = [6,6,6] #number of particles per dimension
    system.dim = 3 #dimension of the sytem (2 or 3 - dimensional)
    system.N = system.n[0]*system.n[1]*system.n[2]  #Number of particles
    system.rho = 0.5 #Number density
    system.L = routines.get_box_dimensions()
    system.alat = system.L[0]/system.n[0] #Lattice parameter
    #system.p = system.L/force.sigma
    system.T = 2. #target temperature (variable with cT "current temp also available")

    # SIMULATIONS VARIABLES
    global DT, iter_equ, iter_prod, rescaling_freq
    DT = 2E-4
    iter_equ = 2000
    iter_prod = 2000
    rescaling_freq = 10


    # SYSTEM CONTAINERS (positions, velocities, ...)
    system.pos = np.zeros((system.N, system.dim), dtype = np.float)
    system.vel = np.zeros((system.N, system.dim), dtype = np.float)
    system.force = np.zeros((system.N, system.dim), dtype = np.float)
    system.f_wall_dw = 0 #force on lower wall
    system.f_wall_up = 0 #force on upper wall
    system.mass = 1 #the particles are assumed to be indentical (not in MQ terms)
    system.time = 0 #probably not very useful

    # SYSTEM INIT ROUTINES
    # These are some of the initial routines for initializing the system,
    # such as lattice positions, random velocities.
    # These routines may vary from simulation to simulation
    routiness.lattice_position()
    routines.vel_random()
    routines.vel_shift()
    routines.vel_rescale(system.T)
    force.LJ_potential_shift()
