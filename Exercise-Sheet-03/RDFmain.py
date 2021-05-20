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
import time
from numba import jit, njit, vectorize # This proved surprisingly usefull for computational efficiency

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
    #printing.openfiles("equilibration")


    iter = 0
    while iter < settings.iter_equ:

        if(iter%200==0):
            print("Equilibration iteration: ",iter)

        # Rescale velocity
        if(iter%settings.rescaling_freq == 0):
            system.vel_rescale(2)

        # Integrate system (if iter = 0 the force needs to be computed
        # integrator module)
        if(iter == 0):
            system.force, system.potential = force.lennard_jones(np.zeros((system.N, system.dim), dtype = np.float), system.pos, system.L, system.N, system.dim)
            #for debugging usual numpy version
            #force.lennard_jones_numpy()
        integrator.velocity_verlet()
        system.compute_energy

        iter+= 1




    ######## PRODUCTION RUN #######

    iter = 0
    while iter < settings.iter_prod:

        if(iter%200==0):
            print("Production iteration: ",iter)

        integrator.velocity_verlet()
        system.compute_energy

        iter+= 1


    printing.print_ovito("after_equilibration.txt")
    rdf, compressibility, coord_number, bins = force.radial_distribution_function(system.pos, system.L, system.N)
    printing.plot_rdf(rdf, bins[1:])

    ######## COOLING RUN #######
    rescaling_temp = np.linspace(system.T,system.T/30,100)
    iter = 0
    t_iter = 0
    while iter < 10000:

        if(iter%200==0):
            print("Cooling iteration: ",iter)

        # Rescale velocity
        if(iter%100 == 0):
            system.vel_rescale(rescaling_temp[t_iter])
            t_iter += 1

        integrator.velocity_verlet()
        system.compute_energy

        iter+= 1




    ######## 2nd PRODUCTION/RELAXATION RUN #######
    iter = 0
    while iter < settings.iter_prod:

        if(iter%200==0):
            print("Production iteration: ",iter)

        integrator.velocity_verlet()
        system.compute_energy

        iter+= 1

    printing.print_ovito("after_cooling.txt")
    rdf, compressibility, coord_number, bins = force.radial_distribution_function(system.pos, system.L, system.N)
    printing.plot_rdf(rdf, bins[1:])
