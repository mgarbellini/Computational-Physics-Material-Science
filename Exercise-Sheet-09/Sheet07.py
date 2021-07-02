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

        if(iter == 0):
            system.force, system.potential, vir = force.lennard_jones(
                np.zeros((system.force.shape), dtype=np.float), system.pos, system.L)

        integrator.nose_hoover_integrate(iter)

def production_run():

    for iter in tqdm(range(0, int(settings.iter_prod)), desc ="Production run T=300K "):

        integrator.nose_hoover_integrate(iter)

        if iter%settings.sampling_freq == 0:
            printing.save_array(system.pos, "positions")


if __name__ == '__main__':

    """System Initialization"""
    settings.init()
    equilibration_run()
    production_run()
