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
import routines
import numpy as np

"""Simulation variables"""
DT = None
m = None
iter_equ = None
iter_prod = None
rescaling_freq = None


def init():
    """Initializes all system variables and runs necessary routines"""

    """TYPE OF ENSEMBLE"""
    system.ensemble = 'micro'


    """LJ POTENTIAL VARIABLES"""
    force.epsilon = 1.
    force.sigma = 1.
    force.cutoff = 4. #in units of sigma


    """SYSTEM VARIABLES"""
    system.n = [4,4,4] #number of particles per dimension
    system.dim = 3 #dimension of the sytem (2 or 3 - dimensional)
    system.N = system.n[0]*system.n[1]*system.n[2]  #Number of particles
    system.rho = 0.5 #Number density
    system.L = routines.get_box_dimensions()
    system.alat = system.L[0]/system.n[0] #Lattice parameter
    #system.p = system.L/force.sigma
    system.T = 2. #target temperature (variable with cT "current temp also available")


    """SIMULATION VARIABLES"""
    global DT, m, iter_equ, iter_prod, rescaling_freq
    DT = 2E-4
    m = 50
    iter_equ = 2000
    iter_prod = 2000
    rescaling_freq = 10


    """SYSTEM/PARTICLES VARIABLES"""
    system.mass = 1 #the particles are assumed have the same mass
    system.pos = np.zeros((system.N, system.dim), dtype = np.float)
    system.vel = np.zeros((system.N, system.dim), dtype = np.float)
    system.force = np.zeros((system.N, system.dim), dtype = np.float)
    system.time = 0 #probably not very useful
    system.xi = 0
    system.lns = 0


    """SYSTEM INIT ROUTINES"""
    routines.lattice_position()
    routines.vel_random()
    routines.vel_shift()
    routines.vel_rescale()
    force.LJ_potential_shift()
    routines.compute_Q()

def check_init():

    print(system.ensemble)


    """LJ POTENTIAL VARIABLES"""
    print(force.epsilon, force.sigma, force.cutoff)



    """SYSTEM VARIABLES"""

    print(system.n, system.dim, system.N, system.rho, system.L, system.alat)
    print(system.T)

    """SIMULATION VARIABLES"""
    print(DT, m, iter_equ, iter_prod, rescaling_freq)

    print(system.mass, system.pos[0,0], system.vel[0,0], system.force[0,0], system.xi, system.lns, system.Q)
