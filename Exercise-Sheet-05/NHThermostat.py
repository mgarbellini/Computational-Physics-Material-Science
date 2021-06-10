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
Latest update: June 7th 2021
"""
# Core namespaces
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from pyfiglet import Figlet
from tqdm import tqdm
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


    CV = []
    E = []
    U = []
    T = []

    ######## EQUILIBRATION RUN #######

    #f = Figlet(font='big') #http://www.figlet.org/examples.html
    #print (f.renderText('Nose-Hoover Thermostat'))

    for iter in tqdm(range(0, settings.iter_equ), desc ="Equilibration run"):
    #for iter in range(2000):
    #for iter in range(0, settings.iter_equ):

        # Rescale velocity
        #if(iter % settings.rescaling_freq == 0):
        #    routines.velocity_rescale()


        # Integrate system (if iter = 0 the force needs to be computed
        if(iter == 0):
            system.force, system.potential = force.lennard_jones(
                np.zeros((system.force.shape), dtype=np.float), system.pos, system.L)

        integrator.nose_hoover_integrate(iter)





    ######## 1st PRODUCTION RUN #######

    for iter in tqdm(range(0, int(settings.iter_prod)), desc ="1st Production run T=300K"):
    #for iter in range(2000):
    #for iter in range(0, settings.iter_prod):
        integrator.nose_hoover_integrate(iter)
        """
        if iter%20==0:
            U.append(system.potential)
            E.append(system.energy)
            T.append(routines.current_temp())
            CV.append(routines.specific_heat(E, U, T))
        """

    #CVE = [item[0] for item in CV]
    #CVU = [item[1] for item in CV]



    """Plots"""
    #printing.plot(None,U,"Potential energy", "Iterations", "Potential U", "Potential Energy over time", "P")
    #printing.plot(None,K,"Total energy", "Iterations", "Energy", "System Energy over time", "Energy")
    #printing.plot(None, CVE,"$CV(\\sigma_E^2)$","Iterations", "Specific Heat", "Specific Heat", "CVE")
    #printing.plot(None, CVU,"$CV(\\sigma_U^2)$","Iterations", "Specific Heat", "Specific Heat", "CVU")
    #printing.plot(None, T, "Temperature", "Iterations", "T", "Temp over time", "T")
