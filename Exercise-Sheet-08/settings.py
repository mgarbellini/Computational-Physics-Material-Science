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
from scipy import special


"""Simulation variables"""
DT = None
m = None
iter_equ = None
iter_prod = None
rescaling_freq = None
sampling_freq = None
ovito_freq = None
kb = 1.38E-23


def init():
    """Initializes all system variables and runs necessary routines"""

    """TYPE OF ENSEMBLE"""
    system.ensemble = 'NHT'
    system.T = 300

    """LJ POTENTIAL VARIABLES"""
    force.epsilon = 0.1*system.T*kb
    force.sigma = 2.55E-10
    force.epsilon_wall = force.epsilon
    force.sigma_wall = force.sigma
    force.cutoff = 5*force.sigma  # in units of sigma
    force.cutoff_wall = 5*force.sigma

    """ELECTROSTATIC PONTENTIAL VARIABLES"""
    force.Rc = 5*force.sigma
    system.surface_charge = 0.005


    """SYSTEM VARIABLES"""
    system.n = [5, 5, 10]  # number of particles per dimension
    system.dim = 3  # dimension of the sytem (2 or 3 - dimensional)
    system.N = system.n[0]*system.n[1]*system.n[2]  # Number of particles
    system.rho = 0.05/force.sigma**3  # Number density
    system.L = routines.get_box_dimensions()
    system.alat = system.L[0]/system.n[0]  # Lattice parameter
    system.p = system.L/force.sigma
    system.V = system.L[0]*system.L[1]*system.L[2]
    system.virial = 0
    # target temperature (variable with cT "current temp also available")


    """SIMULATION VARIABLES"""
    global DT, m, iter_equ, iter_prod, rescaling_freq, sampling_freq, ovito_freq
    DT = 1E-15
    m = 50
    iter_equ = 5000
    iter_prod = 10*iter_equ
    rescaling_freq = 10
    sampling_freq = 10
    ovito_freq = 50

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
    routines.velocity_random()
    routines.get_particle_charge()

    lpos, hpos, system.discrete_surface_q = routines.get_discrete_charge()
    system.discrete_surface_q_pos = np.stack((lpos, hpos), axis = 2)


    system.kinetic = integrator.compute_kinetic(system.vel, system.mass)
    routines.velocity_rescale()
    force.LJ_potential_shift()

    force.c_shift = special.erfc(1)/(force.Rc**2) + np.pi*np.exp(-1)*force.Rc
    routines.compute_Q()
