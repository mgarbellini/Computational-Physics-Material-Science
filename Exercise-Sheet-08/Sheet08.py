#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 07: Structure Factor

*Objective*


*Comments*

"""

# Core namespaces
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from pyfiglet import Figlet
from tqdm import tqdm
from numba import jit, njit, vectorize

# Tailored modules/namespaces
import const  # physical constants
import system  # system descriptions and routine
import force  # force routines
import settings  # simulation input values and initialization
import printing  # various printing and plotting routines
import integrator  # integration scheme routines
import routines # collection of useful routines

def equilibration_run():
    for iter in tqdm(range(0, settings.iter_equ), desc ="Equilibration run T=300K "):

        if iter%settings.ovito_freq == 0: printing.print_ovito(iter, "setup.txt", 'a')


        if(iter == 0):
            system.force, system.potential, vir = force.lennard_jones(
                np.zeros((system.force.shape), dtype=np.float), system.pos, system.L)

            system.force, system.potential = force.lennard_jones_wall(
                system.force, system.pos, system.L, system.potential)

            system.force = force.coulombic_wall(
                system.force, system.pos, system.discrete_surface_q_pos, system.charge, system.discrete_surface_q, system.L)

        integrator.nose_hoover_integrate(iter)

def production_run():

    for iter in tqdm(range(0, int(settings.iter_prod)), desc ="Production run T=300K "):
        if iter%settings.ovito_freq == 0: printing.print_ovito(iter, "setup.txt", 'a')

        integrator.nose_hoover_integrate(iter)

        if iter%settings.sampling_freq == 0:
            printing.save_1d_array(system.pos[:,2], "positions_neg")


if __name__ == '__main__':

    """System Initialization"""
    settings.init()
    printing.print_ovito(0, "setup.txt", 'w')

    equilibration_run()
    production_run()
