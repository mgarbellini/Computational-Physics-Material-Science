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

class System:

    def __init__(self, n_particles):

        self.pos = np.zeros((3, n_particles), dtype = np.float)
        self.vel = np.zeros((3, n_particles), dtype = np.float)
        self.force = np.zeros((3, n_particles), dtype = np.float)
        self.mass = 1 #(float) //the particles are assumed to be indentical (not in MQ terms)

        self.n_particles = n_particles

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
            self.vel = np.random.uniform(-1.0, 1.0, (3, self.n_particles))
        elif(default == 'boltzmann'):
            self.vel = np.random.uniform(-1000, 1000, (3, self.n_particles))

    # Routine for shifting the velocities. This is done in order to cancel
    # the linear momentum of the system.
    # //The mean over each axis is computed and subtracted from the velocity values
    def vel_shift(self):
        mean = np.sum(self.vel, axis = 1)/self.n_particles
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
        self.temp = 2*self.kinetic/(3*BOLTZ*self.n_particles)

    # Routine for rescaling the velocities in order to achieve
    # the target temperature for the system using Eq (7.21)
    # //note: re-computing the temperature might not always be necessary
    def vel_rescale(self):
        self.compute_temperature()
        self.vel = self.vel * np.sqrt(self.target_temp/self.temp)
        self.compute_temperature()


    # Routines for printing useful values
    def print_velocity_sum(self):
        print(np.sum(self.vel, axis = 1))
    def print_velocity(self):
        print(self.vel)


if __name__ == '__main__':

    ensemble = System(4)
    ensemble.print_velocity()
    ensemble.random_velocity()
    ensemble.print_velocity()
    ensemble.shift_velocity()
    ensemble.compute_temperature()
    print(ensemble.temp)
