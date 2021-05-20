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
binwidth = None


# Routine implementing the minimum image convention
# for computing the shortest distance between particles
@njit
def mic(xi, xj, L):

	rij = xi - xj
	if abs(rij) > 0.5*L:
		rij = rij - np.sign(rij) * L

	return rij

@njit
def lennard_jones(force, pos, L, N, dim):
	potential = 0
	rijz = 0

	for i in range(N-1):
		for j in range(i+1, N):
			rijx = mic(pos[i, 0],pos[j,0], L)
			rijy = mic(pos[i, 1],pos[j,1], L)
			if(dim==3):
				rijz = mic(pos[i, 2],pos[j,2], L)
				r = rijx*rijx + rijy*rijy + rijz*rijz
			else:
				r = rijx*rijx + rijy*rijy


			if(r<cutoff*cutoff):
				force[i,0] += 48*epsilon*(sigma**12*rijx/r**7 - 0.5*sigma**6*rijx/r**4)
				force[i,1] += 48*epsilon*(sigma**12*rijy/r**7 - 0.5*sigma**6*rijy/r**4)

				force[j,0] -= 48*epsilon*(sigma**12*rijx/r**7 - 0.5*sigma**6*rijx/r**4)
				force[j,1] -= 48*epsilon*(sigma**12*rijy/r**7 - 0.5*sigma**6*rijy/r**4)

				if(dim==3):
					force[i,2] += 48*epsilon*(sigma**12*rijz/r**7 - 0.5*sigma**6*rijz/r**4)
					force[j,2] -= 48*epsilon*(sigma**12*rijz/r**7 - 0.5*sigma**6*rijz/r**4)

				potential += 2*4*epsilon*(1/r**6 - 1/r**3) - potential_shift

	return force, potential


def radial_distribution_function(pos, L, N):
	n_bins = 100
	#bin_width = L*np.sqrt(3)/2/n_bins
	d = []
	rdf = []
	rijz = 0

	for i in range(N):
		for j in range(N):
			if(i!=j):
				rijx = mic(pos[i, 0],pos[j,0], L)
				rijy = mic(pos[i, 1],pos[j,1], L)
				#rijz = mic(pos[i, 2],pos[j,2], L)
				r = np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz)
				d.append(r)


	max_dist = np.max(np.asarray(d))
	bins = np.linspace(0., max_dist, n_bins)

	delta = bins[1]-bins[0]
	count = 0
	for i in range(len(bins)-1):
		for j in range(len(d)):
			if ((d[j] >= bins[i])&(d[j]< bins[i+1])):
				count +=1
			#shell_volume = 4*np.pi*((bins[i]+delta)**3-bins[i]**3)/3
			shell_area = np.pi*((bins[i]+delta)**2-bins[i]**2)
		#count = count/N/shell_volume/system.rho
		count = count/N/shell_area/system.rho
		rdf.append(count)
		count = 0

	#Integral: coordination number
	coord_number = 4*np.pi*system.rho * np.cumsum(np.asarray(rdf)*delta*bins[1:]**2)

	#Integral: compressibility
	func = np.asarray(rdf) - 1
	compressibility = np.cumsum(1/(system.T) * func * delta) + 1/(system.T  * system.rho)

	return np.asarray(rdf), compressibility, coord_number, bins


def lennard_jones_numpy():

    # (N,N) matrices containing all particles' positions
    X = np.transpose(system.pos[:,0] * np.ones((system.N, system.N)))
    Y = np.transpose(system.pos[:,1] * np.ones((system.N, system.N)))
    Z = np.transpose(system.pos[:,2] * np.ones((system.N, system.N)))

    # Compute "absolute" distance between particles (no PBC and MIC)
    r_x = X - np.transpose(X)
    r_y = Y - np.transpose(Y)
    r_z = Z- np.transpose(Z)

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
    f_x = 4*epsilon*(-12*sigma**12*np.multiply(r_x, np.power(r_reciprocal, 14)) + 6*sigma**6*np.multiply(r_x, np.power(r_reciprocal, 8)))
    f_y = 4*epsilon*(-12*sigma**12*np.multiply(r_y, np.power(r_reciprocal, 14)) + 6*sigma**6*np.multiply(r_y, np.power(r_reciprocal, 8)))
    f_z = 4*epsilon*(-12*sigma**12*np.multiply(r_z, np.power(r_reciprocal, 14)) + 6*sigma**6*np.multiply(r_z, np.power(r_reciprocal, 8)))

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

    # Save potential energy in p_energy variable in py
    system.potential = np.sum(np.triu(P))

def LJ_potential_shift():
    global potential_shift
    potential_shift = 4*epsilon*(np.power(sigma/cutoff, 12) - np.power(sigma/cutoff, 6))
