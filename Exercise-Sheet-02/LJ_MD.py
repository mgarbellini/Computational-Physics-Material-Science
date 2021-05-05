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
A_LATTICE = 1 #Atom lattice interdistance
DT = 1 #Simulation timestep

class System:

    def __init__(self, n):

        # Particles variables (position, velocity, net force and mass)
        self.pos = np.zeros((n**3, 3), dtype = np.float)
        self.vel = np.zeros((n**3, 3), dtype = np.float)
        self.force = np.zeros((n**3, 3), dtype = np.float)
        self.mass = 1 #the particles are assumed to be indentical (not in MQ terms)

        # System variables
        self.N = n**3
        self.L = None
        self.number_density = None
        self.r_cutoff = None
        self.epsilon = None
        self.sigma = None

        # Neighbor variables
        self.neighbor_r_cutoff = None
        self.neighbor_list = None

        # Energy and Thermodynamics variables
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

    # Routine for computing the shortest distance between two given particles,
    # obeying the necessary PBC and implementing the minimum image convention (cfr. notes)
    # //the shortest distance is found by looping over the 3 axis and computing the following
    # // Δx = Δx - L*int(Δx/L)
    def shortest_distance(self, p1, p2):
        d_x = self.pos[p2,0] - self.pos[p1,0]
        d_y = self.pos[p2,1] - self.pos[p1,1]
        d_z = self.pos[p2,2] - self.pos[p1,2]
        return np.sqrt((d_x - self.L*int(d_x/self.L))**2 + (d_y - self.L*int(d_y/self.L))**2 + (d_z - self.L*int(d_z/self.L))**2)

    # Routine for computing the neighbor list of the particles in the system
    # given a specified neighbor-cutoff radius
    def compute_neighbor_list(self):
        self.neighbor_list = []
        for i in range(self.N):
            a_list = []
            for j in range(self.N):
                if i==j:
                    continue
                if self.shortest_distance(i,j) < self.neighbor_r_cutoff:
                    a_list.append(j)

            self.neighbor_list.append(a_list.copy()) #remember that lists are mutable objects, thus appending a copy

    # Routine for computing the force on a given particle
    # //adds a contribution to net_force of the ith and jth particles
    # //note that explicit power calculation, i.e. x*x*...*x is almost double as fast as x**n
    def compute_force(self, p1, p2):
        distance = self.shortest_distance(p1, p2)
        if(distance<self.r_cutoff):
            versor = (self.pos[p1,:] - self.pos[p2, :])/np.linalg.norm(self.pos[p1,:] - self.pos[p2, :])
            force = 48*self.epsilon*self.sigma**12/distance**13 - 24*self.epsilon*self.sigma**6/distance**7
            self.force[p1, :] += -force*versor
            self.force[p2, :] += force*versor

    # Routine for computing the force of the system considering the
    # entire ensemble (only forcing the r_cutoff)
    def compute_force_system(self):
        self.force = np.zeros((self.N, 3), dtype = np.float)
        for p1, p2 in itertools.combinations(np.arange(0,self.N,1), 2):
            self.compute_force(p1,p2)

    # Routine for computing the force of the system using the neighbor_list
    def compute_force_neighbor(self):
        self.force = np.zeros((self.N, 3), dtype = np.float)
        for p1 in range(self.N):
            for p2 in self.neighbor_list[p1]:
                self.compute_force(p1,p2)

    # Routine for evolving the system and calculating trajectories. The algorithm implemented
    # is the known Velocity-Verlet.
    # //the timestep and iterations are taken as arguments
    #
    # The Velocity-Verlet integrator is given by the following equations for position and velocities
    # //note that in practice this equations are vectorial
    # Eq(3) r(t+dt) = r(t) + v(t) * dt + 1/2m * dt^2 * f(t)
    # Eq(4) v(t+dt) = v(t) + dt/2m * [f(t) + f(t+dt)]
    def evolve_system(self, iterations, timestep = 1):
        iter = 0
        while iter < iterations:

            # force computation at current coordinates (timestamp = 0)
            self.compute_force_system()
            force_previous = self.force

            # update system positions
            self.pos += DT*self.vel + 0.5*DT*DT*force_previous/self.mass

            # force computation at new coordinates
            self.compute_force_system()
            self.vel += 0.5*DT*(self.force + force_previous)/self.mass
            iter += 1


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

    ensemble = System(3)
    ensemble.set_number_density(0.1)
    ensemble.distribute_position_cubic_lattice()

    ensemble.neighbor_r_cutoff = 7
    ensemble.r_cutoff = 100
    ensemble.sigma = 1
    ensemble.epsilon = 1
    ensemble.compute_neighbor_list()

    ensemble.evolve_system(5)
