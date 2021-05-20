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

    # SYSTEM VARIABLES
    n = 20 #number of particles per dimension
    system.dim = 2 #dimension of the sytem (2 or 3 - dimensional)
    system.N = n**system.dim  #Number of particles
    system.rho = 0.5 #Number density
    system.L = n*np.power(1/system.rho, 1/system.dim) #Box dimensions (per edge)
    system.alat = system.L/n #Lattice parameter
    system.p = system.L/force.sigma
    system.T = 2. #target temperature (variable with cT "current temp also available")

    # SIMULATIONS VARIABLES
    global DT, iter_equ, iter_prod, rescaling_freq
    DT = 2E-4
    iter_equ = 2000
    iter_prod = 2000
    rescaling_freq = 10

    # PRINTING VARIABLES

    printing.eq_print = 20
    printing.eq_energy_file = "LJMD_" + str(system.N) + "_equil_energy.txt"
    printing.eq_temp_file = "LJMD_" + str(system.N) + "_equil_temperature.txt"
    printing.eq_pos_file = "LJMD_" + str(system.N) + "_equil_positions.txt"
    printing.eq_vel_file = "LJMD_" + str(system.N) + "_equil_velocity.txt"

    printing.prod_print = 20
    printing.prod_energy_file = "LJMD_" + str(system.N) + "_prod_energy.txt"
    printing.prod_temp_file = "LJMD_" + str(system.N) + "_prod_temperature.txt"
    printing.prod_pos_file = "LJMD_" + str(system.N) + "_prod_positions.txt"
    printing.prod_vel_file = "LJMD_" + str(system.N) + "_prod_velocity.txt"


    # SYSTEM CONTAINERS (positions, velocities, ...)
    system.pos = np.zeros((system.N, system.dim), dtype = np.float)
    system.vel = np.zeros((system.N, system.dim), dtype = np.float)
    system.force = np.zeros((system.N, system.dim), dtype = np.float)
    system.mass = 1 #the particles are assumed to be indentical (not in MQ terms)
    system.time = 0 #probably not very useful

    # SYSTEM INIT ROUTINES
    # These are some of the initial routines for initializing the system,
    # such as lattice positions, random velocities.
    # These routines may vary from simulation to simulation
    system.distribute_position_cubic_lattice()
    system.vel_random()
    system.vel_shift()
    system.vel_rescale(system.T)
    force.LJ_potential_shift()
