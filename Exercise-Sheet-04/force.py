#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* FORCE MODULE *
Contains force calculations (and potential energies) using known potentials:
- Gravitational
- Lennard-Jones

Latest update: May 21th 2021
"""

import numpy as np
import system
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize

epsilon = None
epsilon_wall = None
sigma = None
sigma_wall = None
cutoff = None
cutoff_wall = None
potential_shift = None
binwidth = None


# Routine implementing the minimum image convention
# for computing the shortest distance between particles
@njit
def mic(xi, xj, L):

	rij = xi - xj
	if abs(rij) > 0.5 * L:
		rij = rij - np.sign(rij) * L

	return rij


@njit
def lennard_jones(force, pos, L):

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
def lennard_jones_wall(pos, L, force, potential, f_wall_dw, f_wall_up):

	const = 3 * 0.5 * np.sqrt(3) * epsilon_wall * sigma_wall**3
	f_wall_count_dw = 0
	f_wall_count_up = 0
	f_up = 0
	f_dw = 0

	for i in range(pos.shape[0]):
		if(pos[i, 2] < cutoff_wall):
			z = pos[i, 2]
			potential += const * (sigma_wall**6 / z**9 - 1 / z**3)
			force[i, 2] -= const * (-9 * sigma_wall**6 / z**10 + 3 / z**4)

			#wall force
			f_dw += const * (-9 * sigma_wall**6 / z**10 + 3 / z**4)
			f_wall_count_dw += 1

		if(pos[i, 2] > (L[2] - cutoff_wall)):
			z = L[2] - pos[i, 2]
			potential += const * (sigma_wall**6/z**9 - 1/z**3)
			force[i, 2] += const * (-9 * sigma_wall**6/z**10 + 3/z**4)

			#wall force
			f_up -= const * (-9 * sigma_wall**6 / z**10 + 3 / z**4)
			f_wall_count_up += 1

	if(f_wall_count_dw>0):
		f_wall_dw += f_dw/f_wall_count_dw

	if(f_wall_count_up>0):
		f_wall_up += f_up/f_wall_count_up

	return force, potential, f_wall_dw, f_wall_up


@njit
def external_force(force, k):
	for i in range(force.shape[0]):
		force[i, 2] -= k * epsilon / sigma

	return force


def density(axis, n_bins):
	bins = np.linspace(0., system.L[axis], num=n_bins)
	hist = np.histogram(system.pos[:,axis], bins=bins, density=True)[0]
	return hist

def radial_distribution_function(pos, L, N):
	n_bins = 100
	# bin_width = L*np.sqrt(3)/2/n_bins
	d = []
	rdf = []
	rijz = 0

	for i in range(N):
		for j in range(N):
			if(i!=j):
				rijx = mic(pos[i, 0],pos[j,0], L)
				rijy = mic(pos[i, 1],pos[j,1], L)
				# rijz = mic(pos[i, 2],pos[j,2], L)
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
			# shell_volume = 4*np.pi*((bins[i]+delta)**3-bins[i]**3)/3
			shell_area = np.pi*((bins[i]+delta)**2-bins[i]**2)
		# count = count/N/shell_volume/system.rho
		count = count/N/shell_area/system.rho
		rdf.append(count)
		count = 0

	# Integral: coordination number
	coord_number = 4*np.pi*system.rho * np.cumsum(np.asarray(rdf)*delta*bins[1:]**2)

	# Integral: compressibility
	func = np.asarray(rdf) - 1
	compressibility = np.cumsum(1/(system.T) * func * delta) + 1/(system.T  * system.rho)

	return np.asarray(rdf), compressibility, coord_number, bins


def LJ_potential_shift():
    global potential_shift
    potential_shift = 4*epsilon*(np.power(sigma/cutoff, 12) - np.power(sigma/cutoff, 6))
