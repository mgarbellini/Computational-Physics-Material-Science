#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 03 - Radial Distribution function (RDF) of LJ fluid in NVE microcanonical ensemble

*Objective*
The objective of this exercise is to implement a function that calculates the radial distribution
function g(r), the isothermal compressibility (k_T). It is also asked to run a 2D simulation for
a longer time interval than the previous simulation

*Comments*
It is to be noted that the modules used in this simulation are the same as the previous one,
where the new routines are added. The goal is to add all the new routine from week to week.
Latest update: May 12th 2021
"""
# Core namespaces
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
# This proved surprisingly usefull for computational efficiency
from numba import jit, njit, vectorize

# Tailored modules/namespaces
import const  # physical constants
import system  # system descriptions and routine
import force  # force routines
import settings  # simulation input values and initialization
import printing  # various printing and plotting routines
import integrator  # integration scheme routines
import routines


if __name__ == '__main__':
    settings.init()
    #ttime = time.perf_counter()
    ######## EQUILIBRATION RUN #######
    iter = 0
    while iter < 10:

        if(iter % 1 == 0):
            print("Equilibration iteration: ", iter)

        # Rescale velocity
        if(iter % settings.rescaling_freq == 0):
            routines.vel_rescale()

        # Integrate system (if iter = 0 the force needs to be computed
        # integrator module)

        if(iter == 0):
            system.force, system.potential = force.lennard_jones(np.zeros(system.force.shape, dtype=np.float), system.pos, system.L)

        integrator.nose_hoover_integrate()

        iter += 1

    #print(time.perf_counter() - ttime)
