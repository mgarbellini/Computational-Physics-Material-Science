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

DT = None
iter_equ = None
iter_prod = None


# Routine for initializing all variables of the System
def init():

    # SYSTEM VARIABLES
    system.N = 64 #Number of particles
    system.L = None #Box dimensions (per edge)
    system.alat = None #Lattice parameter
    system.rho = 0.1 #Number density
    system.T = 300 #target temperature (variable with cT "current temp also available")

    # LJ POTENTIAL VARIABLES
    force.epsilon = None
    force.sigma = None
    force.cutoff = None

    # SIMULATIONS VARIABLES
    global DT, iter_equ, iter_prod
    DT = 0.5
    iter_equ = 10
    iter_prod = 20

    # PRINTING VARIABLES
    printing.freq = 10 #frequency of data printing (every n timestep)
    printing.energy_file = "LJ_MD_" + str(system.N) + "_energy.txt"
    printing.pos_file = "LJ_MD_" + str(system.N) + "_pos.txt"
    printing.vel_file = "LJ_MD_" + str(system.N) + "_vel.txt"


    # SYSTEM CONTAINERS (positions, velocities, ...)
    system.pos = np.zeros((system.N, 3), dtype = np.float)
    system.vel = np.zeros((system.N, 3), dtype = np.float)
    system.force = np.zeros((system.N, 3), dtype = np.float)
    system.mass = 1 #the particles are assumed to be indentical (not in MQ terms)
    system.time = 0

    # SYSTEM INIT ROUTINES
    # These are some of the initial routines for initializing the system,
    # such as lattice positions, random velocities.
    # These routines may vary from simulation to simulation
    """
    system.distribute_position_cubic_lattice()
    system.vel_random()
    system.vel_shift()
    system.vel_rescale()
    """
