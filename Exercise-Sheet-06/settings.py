#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* SETTINGS MODULE *
Contains all the settings for a given simulation.
At the first call of settings.init() all specified variables
are initialized and available.

Latest update: June 4th 2021
"""

import system
import force
import const
import routines
import integrator
import numpy as np

"""Simulation variables"""
DT = None
m = None
iter_equ = None
iter_prod = None
rescaling_freq = None
sampling_freq = None
kb = 1.38E-23


def init():
    """Initializes all system variables and runs necessary routines"""

    """TYPE OF ENSEMBLE"""
    system.ensemble = 'NHT'
    system.T = 300

    """LJ POTENTIAL VARIABLES"""
    force.epsilon = 0.5*system.T*kb
    force.sigma = 2.55E-10
    force.cutoff = 2.5*force.sigma  # in units of sigma

    """SYSTEM VARIABLES"""
    system.n = [8, 8, 8]  # number of particles per dimension
    system.dim = 3  # dimension of the sytem (2 or 3 - dimensional)
    system.N = system.n[0]*system.n[1]*system.n[2]  # Number of particles
    system.rho = 0.005/force.sigma**3  # Number density
    system.L = routines.get_box_dimensions()
    system.alat = system.L[0]/system.n[0]  # Lattice parameter
    system.p = system.L/force.sigma
    system.V = system.L[0]*system.L[1]*system.L[2]
    system.virial = 0
    # target temperature (variable with cT "current temp also available")


    """SIMULATION VARIABLES"""
    global DT, m, iter_equ, iter_prod, rescaling_freq, sampling_freq
    DT = 1E-15
    m = 50
    iter_equ = 5000
    iter_prod = 45000
    rescaling_freq = 10
    sampling_freq = 50

    """SYSTEM/PARTICLES VARIABLES"""
    system.mass = 105.52E-27  # the particles are assumed have the same mass
    system.pos = np.zeros((system.N, system.dim), dtype=np.float)
    system.vel = np.zeros((system.N, system.dim), dtype=np.float)
    system.force = np.zeros((system.N, system.dim), dtype=np.float)
    system.time = 0  # probably not very useful
    system.xi = 0
    system.lns = 0

    """SYSTEM INIT ROUTINES"""
    routines.position_lattice()
    routines.velocity_random('boltzmann')
    system.kinetic = integrator.compute_kinetic(system.vel, system.mass)
    routines.velocity_rescale()
    force.LJ_potential_shift()
    routines.compute_Q()
