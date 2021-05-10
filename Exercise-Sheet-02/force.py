#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* FORCE MODULE *
Contains force calculations (and potential energies) using known potentials:
- Gravitational
- Lennard-Jones

Latest update: May 7th 2021
"""

import numpy as np
import system
from numba import jit, njit, vectorize

epsilon = None
sigma = None
cutoff = None
potential_shift = None


@njit()
def lennard_jones():

    # (N,N) matrices containing all particles' positions
    X = system.pos[:,0] * np.ones((system.N, system.N))
    Y = system.pos[:,1] * np.ones((system.N, system.N))
    Z = system.pos[:,2] * np.ones((system.N, system.N))

    # Compute "absolute" distance between particles (no PBC and MIC)
    r_x = np.transpose(X) - X
    r_y = np.transpose(Y) - Y
    r_z = np.transpose(Z) - Z

    # Compute shortest distance according to PBC and MIC (minimum image convention)
    r_x = r_x - system.L * np.rint(np.divide(r_x, system.L))
    r_y = r_y - system.L * np.rint(np.divide(r_y, system.L))
    r_z = r_z - system.L * np.rint(np.divide(r_z, system.L))

    # Compute reciprocal of r
    # //I matrix are added and then subtracted in order to avoid divide by zero
    r_reciprocal = np.reciprocal(np.sqrt(r_x**2 + r_y**2 + r_z**2) + np.eye(system.N)) - np.eye(system.N)

    # Exclude distances longer than the cutoff radius
    # by setting r to zero
    r_reciprocal = np.where(r_reciprocal < np.reciprocal(cutoff), r_reciprocal, 0)

    # Compute force with Lennard Jones potential
    # //this evaluation already contains direction information
    # //f_x, f_y, f_z are (N,N) matrices (with zero on the diagonal)
    f_x = 24*epsilon*(2*sigma**12*np.multiply(r_x, np.power(r_reciprocal, 14)) - sigma**6*np.multiply(r_x, np.power(r_reciprocal, 8)))
    f_y = 24*epsilon*(2*sigma**12*np.multiply(r_y, np.power(r_reciprocal, 14)) - sigma**6*np.multiply(r_y, np.power(r_reciprocal, 8)))
    f_z = 24*epsilon*(2*sigma**12*np.multiply(r_z, np.power(r_reciprocal, 14)) - sigma**6*np.multiply(r_z, np.power(r_reciprocal, 8)))

    # Net force on each particle is obtained by summation over the columns
    # //returns forces in array of dimension (N,1)
    F_x = np.sum(f_x, axis = 0)
    F_y = np.sum(f_y, axis = 0)
    F_z = np.sum(f_z, axis = 0)

    # Stack forces in (N,3) array and save in net_force of system
    system.force = np.stack((F_x, F_y, F_z), axis = 1)

    # Compute the potential energy of the system taking advantage of
    # the already computed minimum distance.
    term = sigma*r_reciprocal
    P = 4*epsilon*(np.power(term, 12) - np.power(term, 6)) + potential_shift

    # Save potential energy in p_energy variable in system.py
    system.potential = np.sum(np.triu(P))

# Routine for computing the potential shift due to the potential cutoff
def LJ_potential_shift():
    global potential_shift
    potential_shift = 4*epsilon*(np.power(sigma/cutoff, 12) - np.power(sigma/cutoff, 6))
