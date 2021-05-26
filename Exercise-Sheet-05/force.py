#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* FORCE MODULE *
Contains force calculations (and potential energies) using known potentials:
- Lennard-Jones

Latest update: May 26th 2021
"""

import numpy as np
import system
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize

"""Parameters needed in the force calculations routines"""
epsilon = None
epsilon_wall = None
sigma = None
sigma_wall = None
cutoff = None
cutoff_wall = None
potential_shift = None


@njit
def mic(xi, xj, L):
	"""Computes shortest distance based on minimum-image-convention (MIC).

	Args:
		xi -- position of i particle
		xj -- position of j particle
		L -- dimension of enclosing box

	Returns:
		rij -- the distance between particle i and j

	Note:
		-- the routine runs using Numba @njit decorator for faster run time
	"""
	rij = xi - xj
	if abs(rij) > 0.5 * L:
		rij = rij - np.sign(rij) * L

	return rij

@njit
def lennard_jones(force, pos, L):
	"""Computes the force on all particles given a Lennard-Jones potential

	Args:
		force -- np.zeros([N, 3]) array for force storage (N = number of particles)
		pos -- np.array([N,3]) array of current positions
		L -- dimension of enclosing box

	Parameters:
		sigma -- parameter from the Lennard-Jones potential
		epsilon -- parameter from the Lennard-Jones potential

	Returns:
		force -- np.array([N, 3]) containing the net force on each particle
		potential -- value of the potential energy of the system

	Note:
		-- the routine runs using Numba @njit decorator for faster run time
	"""
	potential = 0
	rij = []
	r = 0

	for i in range(pos.shape[0] - 1):
		for j in range(i + 1, pos.shape[0]):
			for dim in range(pos.shape[1]):
				dist = mic(pos[i, dim], pos[j, dim], L[dim])
				rij.append(dist)
				r += dist * dist

			if(r < cutoff**2):
				for dim in range(pos.shape[1]):
					f = 48 * epsilon * \
					    (sigma**12 * rij[dim] / r**7 - 0.5 * sigma**6 * rij[dim] / r**4)
					force[i, dim] += f
					force[j, dim] -= f

				potential += 2 * 4 * epsilon * (1 / r**6 - 1 / r**3) - potential_shift

	return force, potential

@njit
def lennard_jones_wall(pos, L, force, potential, axis = 2):
	"""Computes contribution to the force on the particles due to the interactions
	with the wall using a 9-3 LJ potential

	Args:
		pos -- np.array([N,3]) array of current positions
		L -- dimension of enclosing box
		force -- np.array([N, 3]) containing forces (N = number of particles)
		potential -- value of potential energy of the system
		axis -- (default = 2) direction of the force, default is z-axis

	Parameters:
		sigma_wall -- parameter from the Lennard-Jones wall-interaction potential
		epsilon_wall -- parameter from the Lennard-Jones wall-interaction potential

	Returns:
		force -- updated array containing the net force of all particles
		potential -- updated value of the potential energy of the system

	Note:
		-- the routine runs using Numba @njit decorator for faster run time
		-- the routine is usually called after lennard_jones() -> force, potential already initialized
	"""
	const = 3 * 0.5 * np.sqrt(3) * epsilon_wall * sigma_wall**3

	for i in range(pos.shape[0]):
		if(pos[i, axis] < cutoff_wall):
			z = pos[i, axis]
			potential += const * (sigma_wall**6 / z**9 - 1 / z**3)
			force[i, axis] -= const * (-9 * sigma_wall**6 / z**10 + 3 / z**4)


		if(pos[i, axis] > (L[axis] - cutoff_wall)):
			z = L[axis] - pos[i, axis]
			potential += const * (sigma_wall**6/z**9 - 1/z**3)
			force[i, axis] += const * (-9 * sigma_wall**6/z**10 + 3/z**4)


	return force, potential

@njit
def external_force(force, k, axis = 2):
	"""Computes contribution to the force on the particles due to an external force
	directed on a given axis (default z-axis)

	Args:
		force -- np.array([N, 3]) containing forces
		k -- parameter for force strenght
		axis -- (default = 2) direction of the force, default is z-axis

	Parameters:
		sigma -- parameter from the Lennard-Jones potential
		epsilon -- parameter from the Lennard-Jones potential

	Returns:
		force -- updated array containing the net force of all particles
	Notes:
		-- the routine runs using Numba @njit decorator for faster run time
	"""
	for i in range(force.shape[0]):
		force[i, axis] -= k * epsilon / sigma

	return force


def density_profile(axis, nbins = 100):
	"""Computes the density profile of the particles over a given axis

	Args:
		axis -- specifies the axis along which to compute the density profile
		n_bins -- (default = 100) number of sampling bins for the histogram

	Returns:
		hist[0] -- histogram values -> y of density profile
		hist[1] -- histogram bins -> x of density profile
	"""
	bins = np.linspace(0., system.L[axis], num=nbins)
	hist = np.histogram(system.pos[:,axis], bins=bins, density=True)
	return hist[0], hist[1]

@njit
def rdf_distances(pos, L, distances):
	"""Computes distances between all particles needed for the radial
	distribution function calculations

	Args:
		pos -- np.array([N,3]) array of current positions
		L -- dimension of enclosing box
		distances -- np.zeros(N(N-1)) array for storing distances

	Returns:
		distances -- np.array(N(N-1)) containing distances squared

	Notes:
		-- the routine runs using Numba @njit decorator for faster run time
	"""
	d_index = 0
	for i in range(pos.shape[0] - 1):
		for j in range(i + 1, pos.shape[0]):
			rx = mic(pos[i, 0],pos[j,0], L[0])
			ry = mic(pos[i, 1],pos[j,1], L[1])
			rz = mic(pos[i, 2],pos[j,2], L[2])
			distances[d_index] = rx*rx + ry*ry + rz*rz
			distances[d_index+1] = distances[d_index]

	return distances

@njit
def rdf_normalize(rdf, bins, N, rho):
	"""Computes radial distribution function normalization

	Args:
		rdf -- np.array(nbins) containing frequencies
		bins -- np.array(nbins) containing bin values
		N -- total number of particles
		rho -- system number density

	Returns:
		rdf -- normalized rdf values

	Notes:
		-- the routine runs using Numba @njit decorator for faster run time
	"""
	for i in range(rdf.shape[0]):
		shell_volume = 4*np.pi*((bins[i+1])**3-bins[i]**3)/3
		rdf[i] = rdf[i]/(N*shell_volume*rho)

	return rdf


def radial_distribution_function(nbins=100):
	"""Computes the radial distribution function of the system, among with
	the coordination number and the isothermal compressibility

	Args:
		nbins -- (default = 100) number of sampling bins for the rdf histogram

	Returns:
		rdf -- np.array(nbins) with the radial distribution function values
		rdf_bins -- np.array(nbins+1) of bins
		n_c -- np.array(nbins) with the coordination number of the system
		k_t -- value of the isothermal compressibility
	"""

	# Array of distances
	dist = np.zeros(system.N*(system.N-1))
	dist = rdf_distances(system.pos, system.L, dist)
	dist = np.sqrt(dist)

	max_dist = 0.5*L[0]
	bins = np.linspace(0., max_dist, nbins)

	# Radial Distribution Function
	histogram = np.histogram(dist, bins=bins, density=False)
	rdf = rdf_normalize(histogram[0], histogram[1], system.N, system.rho)
	rdf_bins = histogram[1]


	# Coordination Number
	n_c = 4*np.pi*system.rho * np.cumsum(rdf*bins[1]*bins[1:]**2)

	# Isothermal Compressibility
	k_t = np.cumsum(4/(system.T) * np.pi * (rdf-1) * bins[1] * bins[1:]**2) + 1/(system.T  * system.rho)

	return rdf, rdf_bins, n_c, k_t[-1]


def LJ_potential_shift():
	"""Computes the LJ potential shift

	Parameters:
		cutoff -- radius at which the potential is truncated and shifted

	Notes:
		-- this is done once at the beginning of each simulation
	"""

    global potential_shift
    potential_shift = 4*epsilon*(np.power(sigma/cutoff, 12) - np.power(sigma/cutoff, 6))
