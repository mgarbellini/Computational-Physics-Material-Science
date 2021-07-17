#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 08: Bond Potential

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

def distance(pos):

    distances = []
    for m in range(system.mask.shape[0]):
        i_index = system.mask[m,0]
        j_index = system.mask[m,1]

        rx = pos[i_index,0] - pos[j_index,0]
        ry = pos[i_index,1] - pos[j_index,1]
        rz = pos[i_index,2] - pos[j_index,2]

        r = rx*rx + ry*ry + rz*rz
        distances.append(r)

    average = np.mean(np.asarray(distances))
    return average




def equilibration_run():

    for iter in tqdm(range(0, settings.iter_equ), desc ="Equilibration run T=300K "):

        if(iter == 0):
            system.force = force.lennard_jones_bond(
                system.mask, np.zeros((system.force.shape), dtype=np.float), system.pos, system.L)

            system.force = force.bond_potential(system.mask, system.pos, system.force)

        integrator.nose_hoover_integrate(iter)

def production_run():

    bond_len = []
    for iter in tqdm(range(0, int(settings.iter_prod)), desc ="Production run T=300K "):

        integrator.nose_hoover_integrate(iter)

        if iter%settings.sampling_freq == 0:
            printing.save_array(system.pos, "positions")

        if iter%50 == 0:
            bond_len.append(distance(system.pos))

    return bond_len

if __name__ == '__main__':

    """System Initialization"""
    settings.init()



    equilibration_run()
    bond_len = production_run()

    x = np.linspace(0, settings.iter_prod, num=len(bond_len))
    printing.plot(False, x, bond_len, "$Average$", "Iterations", "Bond lenght [m]", "Average bond separation", "Bond")
