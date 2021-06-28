#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* PROFILES SCRIPT *
Script for the density profiles analysis needed in exercise sheet 08.
This allows to do the analysis at a later time

Latest update: June 26th 2021
"""

import numpy as np
from tqdm import tqdm
import routines
import system
import printing
import force
import settings
import matplotlib.pyplot as plt
from numba import njit


def import_array(filename):
    """Loads z axis position from file, returns
    array of positions over the whole production run with shape
    (N-particles, N-iterations)"""

    array = np.loadtxt(filename + ".txt")

    return np.transpose(array)

def density_profile(array,nbins = 100):
    """Computes density profile of given array using np.histogram,
    and returns histogram and bins"""

    bins = np.linspace(0., system.L[2]/force.sigma, num=nbins)
    hist = np.histogram(array/force.sigma, bins=bins, density=True)
    bins = hist[1] - hist[1][1]/2
    return hist[0], bins[1:]

def theoretical_prediction(density, x):
    nm = density[int(len(density)/2)]
    k = np.sqrt(nm*2*np.pi*system.e*system.e/force.epsilon_not/settings.kb/system.T)
    print(k)
    cos2 = np.cos(k*x*force.sigma)**2
    f = float(nm)/cos2
    return f


if __name__ == '__main__':

    settings.init()

    pos = import_array("positions_pos")

    profile = 0
    for i in tqdm(range(pos.shape[1]), desc="Density Profile"):

        p, bins = density_profile(pos[:,i])
        profile += p

    profile = profile/pos.shape[1]
    prediction = theoretical_prediction(profile, bins)

    printing.plot(False, bins, profile, "$n(z)$", "$z$ [$\\sigma$]", "$n(z)$", "Density profile", "densityprofile_simul")
    printing.plot(False, bins, prediction, "$n(z)$", "$z$ [$\\sigma$]", "$n(z)$", "Density profile", "densityprofile_theo")
