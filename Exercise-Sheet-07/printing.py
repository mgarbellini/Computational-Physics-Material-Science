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


def plot(double, x, y, legend, xlabel, ylabel, title, filename):
    """Plots the given input data (can be multidimensional)

    Args:
        x -- x values
        y -- [y1,y2] values
        legend -- legend text description
        xlabel -- x axis label
        ylabel -- y axis label
        title -- plot title
        filename -- output filename

    """
    if double == True:
        y1 = np.asarray(y[0])
        y2 = np.asarray(y[1])
    else:
        y1 = np.asarray(y)

    if x.any() == None:
        x = np.linspace(1, y1.shape[0], num=y1.shape[0], endpoint=True)
    elif not isinstance(x, np.ndarray):
        x = np.asarray(arrray)



    with plt.style.context(['science']):

        fig, ax = plt.subplots()

        if double == True:
            ax.plot(x,y1,label = legend[0])
            ax.plot(x,y2,label = legend[1])
        else:
            ax.plot(x,y1,label = legend)

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

def save_data(data,filename):
    """Saves data to txt file

    Args:
        data -- list containing all data to be saved to file (needs to be unpacked)
        filename -- name of file
    """

    filename = filename + ".txt"
    file = open(filename, 'a')

    for value in data:
        file.write(str(value) + " ")

    file.write("\n")
    file.close()

def save_array(array, filename):
    """Saves numpy array to file

    Args:
        array -- array containing data to be saved
        filename -- name of file
    """

    filename = filename + '.txt'
    file = open(filename, 'a')

    for i in range(array.shape[0]):
        for dim in range(array.shape[1]):
            file.write(str(array[i,dim]) + " ")

    file.write("\n")
    file.close()
