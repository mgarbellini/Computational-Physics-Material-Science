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

        if iter%100 == 0:
            values, bins = routines.radial_distribution_function()
            if iter == 0:
                RDF = values
            else:
                RDF = RDF + values

        if iter%settings.sampling_freq == 0:
            U.append(system.potential)
            E.append(system.energy)
            T.append(routines.current_temp())
            cv = routines.specific_heat(E, U, T)
            CV.append(cv)



    RDF = RDF/settings.iter_prod*100
    printing.plot(False, bins, RDF, "$g(r)$", "$r$ [m]", "$g(r)$", "Radial distribution function", "rdf")

    """
    CVE = [item[0] for item in CV]
    CVU = [item[1] for item in CV]

    printing.plot(False, None, E, "$E = K + U + E_{NH}$", "Iterations", "Total Energy [J]", "Total Energy over time", "E")
    printing.plot(False, None, U, "U", "Iterations", "Potential Energy [J]", "Potential Energy over time", "P")
    printing.plot(True, None, [CVE, CVU], ["Cv with $\\sigma_E^2$","Cv with $\\sigma_U^2$"], "Iterations", "Specific Heat", "Specific Heat over time", "CV")
    """
