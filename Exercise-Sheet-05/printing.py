#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* PRINTING MODULE *
Contains all the routines for printing and saving computational
data to file. This includes basic plotting and ovito style outputs,
as well as printing routines for debugging

Latest update: May 27th 2021
"""

import sys
import time
import settings
import numpy as np
import system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # PLOTTING  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plot(x, y, legend, xlabel, ylabel, title, filename):
    """Plots the given input data (can be multidimensional)

    Args:
        x -- x values
        y -- y values
        legend -- legend text description
        xlabel -- x axis label
        ylabel -- y axis label
        title -- plot title
        filename -- output filename

    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    if x == None:
        x = np.linspace(1, y.shape[0], num=y.shape[0], endpoint=True)
    elif not isinstance(x, np.ndarray):
        x = np.asarray(arrray)


    with plt.style.context(['science']):

        fig, ax = plt.subplots()
        for i in range(y.ndim):
            if y.ndim == 1:
                ax.plot(x,y,label = legend)
            else:
                ax.plot(x, y[:,i], label = legend[i])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(title)
        fig.savefig('./' + filename +'.pdf')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # PRINTING TO FILE  # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def print_ovito(filename):
    ovito = open(filename, 'w')
    ovito.write("ITEM: TIMESTEP \n")
    ovito.write("%i \n" % 0)
    ovito.write("ITEM: NUMBER OF ATOMS \n")
    ovito.write("%i \n" % system.N)
    ovito.write("ITEM: BOX BOUNDS pp pp pp \n")
    ovito.write("%e %e \n" % (system.L[0], system.L[0]))
    ovito.write("%e %e \n" % (system.L[1], system.L[1]))
    ovito.write("%e %e \n" % (system.L[2], system.L[2]))
    ovito.write("ITEM: ATOMS id x y z \n")
    for i in range(0, system.N):
        ovito.write("%i %e %e %e \n" %
                    (i, system.pos[i, 0], system.pos[i, 1], system.pos[i, 2]))

    ovito.close()
