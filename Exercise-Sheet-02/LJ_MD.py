#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 02 - Lennard-Jones fluid in microcanonical enesemble

*Objective*
Implement a molecular dynamics (MD) code that computes the trajectories of
a collection of N particles interacting through the Lennard-Jones (LJ)
potential in an (NVE) microcanonical ensemble.

"""
import numpy as np
import sys
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BOLTZ = 1 #Boltzmann constant Kb
A_LATTICE = 1

class System:

    def __init__(self, n):

        self.pos = np.zeros((n**3, 3), dtype = np.float)
        self.vel = np.zeros((n**3, 3), dtype = np.float)
        self.force = np.zeros((n**3, 3), dtype = np.float)
        self.mass = 1 #(float) //the particles are assumed to be indentical (not in MQ terms)

        self.N = n**3
        self.L = None
        self.number_density = None

        self.u_energy = None
        self.kinetic = None
        self.potential = None
        self.target_temp = None
        self.temp = None

    # Routine for generating random velocities
    # //default is uniform distribution
    # //boltzmann distribution needs to be implemented
    def vel_random(self, default = 'uniform'):
        if(default == 'uniform'):
            self.vel = np.random.uniform(-1.0, 1.0, (3, self.N))
        elif(default == 'boltzmann'):
            self.vel = np.random.uniform(-1000, 1000, (3, self.N))

    # Routine for shifting the velocities. This is done in order to cancel
    # the linear momentum of the system.
    # //The mean over each axis is computed and subtracted from the velocity values
    def vel_shift(self):
        mean = np.sum(self.vel, axis = 1)/self.N
        self.vel[0,:] -= mean[0]
        self.vel[1,:] -= mean[1]
        self.vel[2,:] -= mean[2]

    # Routine for calculating the kinetic energy of the system
    def compute_kinetic(self):
        self.kinetic = 0.5*self.mass*np.sum(np.multiply(self.vel, self.vel))

    # Routine for computing the termperature of the system following Eq (7.19) of
    # the lectures script.
    # //for the time being this is achieved by explicitly evaluating the
    # //kinetic energy of the system. In a later time for precision purposes calculating
    # //the potential energy could be recommended
    def compute_temperature(self):
        self.compute_kinetic()
        self.temp = 2*self.kinetic/(3*BOLTZ*self.N)

    # Routine for rescaling the velocities in order to achieve
    # the target temperature for the system using Eq (7.21)
    # //note: re-computing the temperature might not always be necessary
    def vel_rescale(self):
        self.compute_temperature()
        self.vel = self.vel * np.sqrt(self.target_temp/self.temp)
        self.compute_temperature()

    # Routine for placing particles on a cubic lattice of given parameter a_lattice.
    # Let us assume for simplicity that the relationship between a_lattice and L (box dim)
    # is given by L
    # //the default a_lattice is defined globally
    def distribute_position_cubic_lattice(self, a_lat = A_LATTICE):
        if(self.L == None):
            if(self.number_density == None):
                print("Error: unspecified number density and box volume. Unable to distribute positions over cubic lattice")
                sys.exit()
            else:
                self.L = np.cbrt(self.N/self.number_density)

        # generates position over a cubic lattice of a_lat = 1 (only positive positions)
        S_range = list(range(0,int(np.cbrt(self.N))))
        cubic_lattice = np.array(list(itertools.product(S_range, repeat=3)))

        # rescaling and shifting
        self.pos = cubic_lattice * (self.L/np.cbrt(self.N)) + (self.L/(2*np.cbrt(self.N)))

    # Routine for plotting the inital lattice positions
    # //works for any position
    def plot_positions(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], s = 100)
        ax.set_xlim([0,self.L])
        ax.set_ylim([0,self.L])
        ax.set_zlim([0,self.L])
        plt.show()

    # Routine for computing the shortest distance between two given two particles,
    # obeying the necessary PBC and implementing the minimum image convention (cfr. notes)
    # //the best way is to create a (NxN) matrix for x,y and z. This allows to return
    def shortest_distance(self):



    # Set of routines for setting system variables. This is useful since it allows for a
    # very general implementation (i.e. no need to set variables in __init__)
    # Some include:
    # - number density
    # - volume (or lenght L of cubic volume)
    # - system (target) temperature
    def set_number_density(self, number_density):
        self.number_density = number_density
    def set_L(self, lenght):
        self.L = lenght
    def set_target_temperature(self, temperature):
        self.target_temp = temperature


    # Set of routines for printing useful values
    def print_velocity_sum(self):
        print(np.sum(self.vel, axis = 1))
    def print_velocity(self):
        print(self.vel)


if __name__ == '__main__':

    ensemble = System(4)
    ensemble.set_number_density(0.1)
    ensemble.distribute_position_cubic_lattice()
    ensemble.plot_positions()
