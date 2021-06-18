#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 06

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
import routines

def equilibration_run():

    for iter in tqdm(range(0, settings.iter_equ), desc ="Equilibration run T=300K "):

        if(iter == 0):
            system.force, system.potential, vir = force.lennard_jones(
                np.zeros((system.force.shape), dtype=np.float), system.pos, system.L)

        integrator.nose_hoover_integrate(iter)

def production_run():

    radial_distribution = 0
    T = []
    P = []
    for iter in tqdm(range(0, int(settings.iter_prod)), desc ="Production run T=300K "):

        integrator.nose_hoover_integrate(iter)

        if iter%settings.sampling_freq == 0:

            T.append(routines.current_temp())
            P.append(routines.current_pressure(system.virial))

            rdf, bins = routines.radial_distribution_function()
            radial_distribution += rdf

    return radial_distribution/len(T), bins, T, P


if __name__ == '__main__':

    """System Initialization"""
    settings.init()
    equilibration_run()
    rdf, bins, T, P = production_run()



    """Plot Temperature and Pressure"""
    x = np.linspace(0, settings.iter_prod, num=len(T))
    printing.plot(False, x, T, "$T(t)$", "Iterations", "Temperature [K]", "Temperature over time", "T")
    printing.plot(False, x, P, "$P(t)$", "Iterations", "Pressure [?]", "Pressure over time", "P")
    #printing.plot(False, x, KT, "$K_T$", "Iterations", "Isothermal Compressibility", "Compressibility $K_T$", "KT")
    printing.plot(False, bins, rdf, "$g(r)$", "$r$ [$\\sigma$]", "$g(r)$", "RDF with $\\rho = 0.005\\sigma^{-3}$", "RDF")
