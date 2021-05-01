"""
M. Garbellini
matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 01 - Planetary Evolution

The following are the referenced equations in the code:
Eq (1) : Euler algorithm for r(t+dt) and v(t+dt)
Eq (2) : Verlet algorithm for r(t+dt) and v(t)
Eq (3) : Velocity-Verlet algorithm for r(t+dt) and v(t+dt)
Please see the report for an in depth formulation.

"""

"""
!% position and velocity data of all 9 planets in the solar system with Sun at origin
!% position in units of AU ( 1AU = 1.49597870691E+11 meter )
!% velocity in units of AU/day ( 1day = 86400 second )
"""

import numpy as np
import sys
import itertools
import time
import matplotlib.pyplot as plt


GRAVITATIONAL_CONST = 6.67408E-11
UNITS_OF_MASS = 10E29 #"""units of mass in kg"""
ASTRO_POS = 1.49597870691E+11
ASTRO_DAY = 86400
TIMESTEP = 1000

planet1_x = []
planet1_y = []
planet2_x = []
planet2_y = []

def load_planets():

    position = np.loadtxt("planets.txt", usecols=(0,1,2))
    velocity = np.loadtxt("planets.txt", usecols=(3,4,5))
    names = np.loadtxt("planets.txt", dtype=object, usecols=(6) )
    mass = np.loadtxt("mass.txt", usecols=(0))

    return names, mass, position, velocity


class System:

    def __init__(self, name, mass):

        self.planet_name = name
        self.t = 0
        self.n_planets = len(self.planet_name)

        self.p_mass = np.resize(mass, (len(mass), len(mass)))
        self.p_mass_matrix = None

        self.p_pos_previous = None
        self.p_vel_previous = None

        self.p_pos_current = None
        self.p_vel_current = None

        self.p_pos_next = None
        self.p_vel_next = None


        self.total_energy = None
        self.potential_energy = None
        self.kinetic_energy = None


    def set_p_mass_matrix(self):
        mass = self.p_mass
        multiplied_mass = np.multiply(mass, np.transpose(mass))
        np.fill_diagonal(multiplied_mass,0)
        self.p_mass_matrix = multiplied_mass


    def set_initial_condition(self, pos, velocity):
        if(self.t != 0):
            print("Error: setting intial condition for t>0")
        position = pos.transpose()
        dim = len(position[0,:])
        self.p_pos_current = np.stack(( np.resize(position[0,:], (dim, dim)),
                                        np.resize(position[1,:], (dim, dim)),
                                        np.resize(position[2,:], (dim, dim))),
                                        axis=0)
        self.p_vel_current = velocity.transpose()
        self.set_p_mass_matrix()


    def net_force(self, pos):

        distance = np.abs(pos - pos.transpose((0, 2, 1)))
        distance_non_null = np.where(distance == 0, 1, distance)
        distance_square_matrix = np.reciprocal(distance_non_null**2)
        mass_stack = np.stack((self.p_mass_matrix, self.p_mass_matrix, self.p_mass_matrix),axis=0)
        net_force_matrix = GRAVITATIONAL_CONST*np.multiply(distance_square_matrix, mass_stack)
        return np.sum(net_force_matrix, axis=1)

    def update_current(self, next_pos, next_vel):

        self.p_pos_previous = self.p_pos_current
        self.p_vel_previous = self.p_vel_current

        self.p_pos_current = np.stack((next_pos, next_pos, next_pos), axis=0)
        dim = len(next_pos[0,:])
        self.p_pos_current = np.stack(( np.resize(next_pos, (dim, dim)),
                                        np.resize(next_pos, (dim, dim)),
                                        np.resize(next_pos, (dim, dim))),
                                        axis=0)
        self.p_vel_current = next_vel

        self.p_pos_next = None
        self.p_vel_next = None

    def evolve_euler(self,iterations):
        iter =0

        while iter<iterations:

            next_pos = self.p_pos_current[:,:,0] + TIMESTEP*self.p_vel_current + 0.5*np.multiply(np.reciprocal(self.p_mass[0:3,:]),self.net_force(self.p_pos_current))*TIMESTEP**2
            next_vel = self.p_vel_current + np.multiply(np.reciprocal(self.p_mass[0:3,:]),self.net_force(self.p_pos_current))*TIMESTEP
            self.update_current(next_pos, next_vel)
            iter+=1

            planet1_x.append(next_pos[0,6])
            planet1_y.append(next_pos[1,6])
            planet2_x.append(next_pos[0,7])
            planet2_y.append(next_pos[1,7])




    #def k_energy(self, vel):

    #def u_energy(self, pos):




if __name__ == '__main__':

    planets_name, planets_mass, planets_pos, planets_vel = load_planets()

    SolarSystem = System(planets_name, planets_mass)
    SolarSystem.set_initial_condition(planets_pos, planets_vel)

    SolarSystem.evolve_euler(10000)

    with plt.style.context(['science', 'dark_background']):
        fig, ax = plt.subplots()

        ax.plot(planet1_x, planet1_y, label="planet1")
        ax.plot(planet2_x, planet2_y, label="planet2")

        ax.legend(title='Planets')
        ax.autoscale(tight=True)
        fig.savefig('./fig1.pdf')
