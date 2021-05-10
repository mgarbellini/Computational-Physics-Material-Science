#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 02 - Lennard-Jones fluid in microcanonical enesemble

*Objective*
Implement a molecular dynamics (MD) code that computes the trajectories of
a collection of N particles interacting through the Lennard-Jones (LJ)
potential in an (NVE) microcanonical ensemble.

Latest update: May 8th 2021
"""
# Core namespaces
import numpy as np
import sys
import time
from numba import jit, njit, vectorize

# Tailored modules/namespaces
import const # physical constants
import system # system descriptions and routine
import force # force routines
import settings # simulation input values and initialization
import printing # various printing and plotting routines
import integrator # integration scheme routines


if __name__ == '__main__':
    settings.init()

    ttime = time.perf_counter()
    ######## EQUILIBRATION RUN #######
    printing.openfiles("equilibration")

    plot_iter = 0
    iter = 1
    while iter < settings.iter_equ:
        print("Equilibration iteration: ",iter)

        # Rescale velocity
        if(iter%settings.rescaling_freq == 0):
            print("Velocity rescaling")
            system.vel_rescale()

        # Integrate system (if iter = 0 the force needs to be computed
        # integrator module)
        if(iter == 0):
            force.lennard_jones()
        integrator.velocity_verlet()
        system.compute_energy()

        # Printing routine
        if(iter%printing.eq_print==0):
            printing.print_system("equilibration")
        iter+= 1
        plot_iter+=1

    printing.closefiles("equilibration")

    ######## PRODUCTION RUN #######
    printing.openfiles("production")

    iter = 0
    while iter < settings.iter_prod:
        print("Production iteration: ",iter)

        integrator.velocity_verlet()
        system.compute_energy()

        # Printing routine
        if(iter%printing.prod_print == 0):
            printing.print_system("production")

        iter+= 1
        plot_iter+=1

    print(time.perf_counter()-ttime)

    printing.closefiles("production")
