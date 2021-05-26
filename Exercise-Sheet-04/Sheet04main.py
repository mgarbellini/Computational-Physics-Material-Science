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


if __name__ == '__main__':
    settings.init()

    printing.print_ovito("initial.txt")

    ttime = time.perf_counter()
    ######## EQUILIBRATION RUN #######
    iter = 0
    while iter < settings.iter_equ:

        if(iter % 200 == 0):
            print("Equilibration iteration: ", iter)

        # Rescale velocity
        if(iter % settings.rescaling_freq == 0):
            system.vel_rescale(2)

        # Integrate system (if iter = 0 the force needs to be computed
        # integrator module)

        if(iter == 0):
            rho_x = 0
            rho_y = 0
            rho_z = 0
            system.force, system.potential = force.lennard_jones(
                np.zeros(system.pos.shape, dtype=np.float), system.pos, system.L)
            system.force, system.potential, system.f_wall_dw, system.f_wall_up = force.lennard_jones_wall(
                system.pos, system.L, system.force, system.potential, system.f_wall_dw, system.f_wall_up)
            system.force = force.external_force(system.force, 1)

        integrator.velocity_verlet()
        system.compute_energy

        iter += 1

    ######## PRODUCTION RUN #######

    iter = 0
    while iter < settings.iter_prod:

        if(iter % 200 == 0):
            print("Production iteration: ", iter)

        integrator.velocity_verlet()
        system.compute_energy

        # get density
        rho_x += force.density(0, 100)
        rho_y += force.density(1, 100)
        rho_z += force.density(2, 200)

        iter += 1

    # average densities and pressure
    rho_x = rho_x / settings.iter_prod
    rho_y = rho_y / settings.iter_prod
    rho_z = rho_z / settings.iter_prod

    pressure_up = system.f_wall_up / \
        (settings.iter_prod + settings.iter_equ) / (system.L[0] * system.L[1])
    pressure_dw = system.f_wall_dw / \
        (settings.iter_prod + settings.iter_equ) / (system.L[0] * system.L[1])
    print(pressure_dw, pressure_up)

    printing.print_ovito("production.txt")

    # plot radial distribution function
    with plt.style.context(['science']):
        bins = np.linspace(0., system.L[2], num=200)
        fig, ax = plt.subplots()
        ax.plot(bins[1:], rho_z, label = r"$\rho(z)$")
        # ax.autoscale(tight=True)
        ax.set_xlabel(r'z $[\sigma]$')
        ax.set_ylabel(r'$\rho(z)$')
        ax.set_ylim([0,2.5])
        ax.legend()
        ax.set_title("Density profile (z axis)")
        fig.savefig('./rhoz.pdf')

    with plt.style.context(['science']):
        binsx = np.linspace(0., system.L[0], num=100)
        fig, ax = plt.subplots()
        ax.plot(binsx[1:], rho_y, label = r"$\rho(y)$")
        ax.plot(binsx[1:], rho_x, label = r"$\rho(x)$")
        # ax.autoscale(tight=True)
        ax.set_xlabel('x/y $[\sigma]$')
        ax.set_ylabel(r'$\rho$')
        #ax.set_ylim([0,])
        ax.legend()
        ax.set_title("Density Profile (x,y axis)")
        fig.savefig('./rhoxy.pdf')
