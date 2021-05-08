#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 01 - Planetary Evolution

The following are the referenced equations in the code:
Eq(1), Eq(2) : Euler algorithm for r(t+dt) and v(t+dt)
Eq(3), Eq(3) : Verlet algorithm for r(t+dt) and v(t)
Eq(4), Eq(5) : Velocity-Verlet algorithm for r(t+dt) and v(t+dt)
Please see the report for an in-depth formulation.

The code consists in two main classes: PARTICLE and SYSTEM. Planets are modelled as particles
and the Solar System as a system. The class particle contains all the informations about a given
particle (mass, position, velocity, net_force). The class system contains all the particles and
meaningful informations about the system, such as potential and kinetic energy of the whole
ensemble of particles.
"""

import numpy as np
import sys
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global constant definition
# //DT simulation timestep (days)
DT = 0.5
# //number of iterations (default is 500 years)
ITERATIONS = 2*365*10

# //the following are constant and conversions useful
# //for astronomical simulations (e.g. normalized Gravitational constant)
AU = 1.49597870691E+11
AUday = 1.49597870691E+11/86400.
G_CONST = 6.67408E-11/(AUday**2)/AU
UNITS_OF_MASS = 1E29 #mass in kg


# Routine for loading planets mass and initial conditions from file
# //note that the initial files mass.dat and planets.dat have been re-formatted
def load_planets():

    position = np.loadtxt("planets.txt", usecols=(0,1,2))
    velocity = np.loadtxt("planets.txt", usecols=(3,4,5))
    names = np.loadtxt("planets.txt", dtype=object, usecols=(6) )
    mass = np.loadtxt("mass.txt", usecols=(0))

    return names, mass, position, velocity


# Particle CLASS. Contains all the information on a given particle
# //practical for large number of particles
class Particle:

    def __init__(self,ID, name, mass, position, velocity):

        # particle ID: useful if index is more significant than the particle name
        # particle name: relevant only for planets
        # particle mass
        self.ID = ID
        self.name = name
        self.mass = mass

        # position and velocity arrays
        # position array is of shape (3,3) and contains the information
        # on x(t-dt), x(t), x(t+dt), accessible respectively by the indices {-1,0,1}
        self._position = np.stack((position, np.zeros(3), np.zeros(3)), axis=0)
        self._velocity = velocity

        # net force on particle
        # additional force array needed for the velocity-verlet scheme
        self.net_force = np.zeros(3,dtype=np.float)
        self.v_verlet_net_force = None


    # Set of routines for setting and accessing _position and _velocity
    # //position requires the timestamp {-1,0,1}
    def pos(self, timestamp):
        return self._position[timestamp,:]

    def vel(self):
        return self._velocity

    def set_vel(self, velocity):
        self._velocity = velocity

    def set_pos(self, position, timestamp):
        self._position[timestamp,:] = position


    # Routine for updating the coordinate, i.e. shifting new->current, current->previous
    # //this function is called by the 'parent' System.update_coordinates after each iteration
    # //net_force is set to zero after each iteration
    def update_coord(self):
        self._position = np.stack((self.pos(1),np.zeros(3),self.pos(0)), axis=0)
        self.net_force = np.zeros(3,dtype=np.float)


    # Routine for printing on standard output the current coordinates
    # //the values are printed with precision to the 5th digit (by default)
    # //although can be overwritten by passing precision argument
    def print_current_coordinates(self, precision = 5):
        print(np.around(self.pos(0)[0], precision),np.around(self.pos(0)[1],precision), np.around(self.pos(0)[2],precision))


# System CLASS. Contains all the particles of a given system. Implements as methods all the various
# integrator, and includes routines for energy evaluation .
class System:

    def __init__(self, name):

        # system ID: useful if more system are needed in the same simulation
        # planets: list containing all the planets
        # time: by accessing self.time the time elapsed (in days) is given
        self.ID = name
        self.planets = []
        self.time = 0

        # the energy values are instantaneous values
        # //history values are not saved on memory
        self.k_energy = None
        self.u_energy = None


    # Routine for adding the system planets
    def add_planet(self,ID, name, mass, position, velocity):
        self.planets.append(Particle(ID,name, mass, position, velocity))


    # Routine for updating the planets coordinates. It is foundamental for optimal handling
    # of positions and velocities at different timestamp (t-dt, t, t+dt)
    # //the routine itself calls a routine implemented in the class particle
    # //the sun coordinates are not updated. It is assumed stationary (sun planet_ID = 0)
    def update_coordinates(self):
        for planet in self.planets:
            if(planet.ID > 0):
                planet.update_coord()


    # Routine for computing the force between two planets
    # //the function adds the force contribution for both planet_1 and planet_2 (saves a little computational time)
    # //the function is called by compute_system_force which computes the force for all planets
    # //F_12 = G m_1 m_2 (r1-r2)/(|r1-r2|)^3
    def compute_force(self, planet_1, planet_2, t_stamp):

        distance_cubed = (np.linalg.norm(planet_1.pos(t_stamp) - planet_2.pos(t_stamp)))**3
        force = G_CONST*planet_1.mass*planet_2.mass * np.reciprocal(distance_cubed) * (planet_1.pos(t_stamp) - planet_2.pos(t_stamp))
        planet_1.net_force += -force
        planet_2.net_force += +force


    # Routine for computing the force between all planets in the system
    # //t_stamp {-1,0,1} represents the position stored at t-dt, t and t+dt
    # //itertools_combinations() returns all the combination of planets
    # //avoiding duplicates (e.g. AB and BA) and self-combinations (e.g. AA, BB)
    def compute_system_force(self, t_stamp):
        for planet_1, planet_2 in itertools.combinations(self.planets, 2):
            self.compute_force(planet_1, planet_2, t_stamp)


    # Routine for computing the kinetic energy of the system (evaluated at timestamp t)
    # //no need to store k_energy for each individual planet
    def compute_k_energy(self):

        # kinetic energy is set to zero each time it is required
        # to avoid unexpected values
        self.k_energy = 0
        for planet in self.planets:
            self.k_energy += 0.5*planet.mass*np.linalg.norm(planet.vel())


    # Routine for computing the potential energy of the system (evaluated at timestamp t)
    # //as for compute_system_force the routine is inefficient for large number of particles
    def compute_u_energy(self):

        # potential energy is set to zero each time it is required
        # to avoid unexpected values
        self.u_energy = 0
        for planet_1, planet_2 in itertools.combinations(self.planets, 2):
            distance = np.linalg.norm(planet_1.pos(0) - planet_2.pos(0))
            self.u_energy += -G_CONST*planet_1.mass*planet_2.mass * np.reciprocal(distance)


    # Routine for printing on standard output the current coordinates and the system energies
    # //the values are printed with precision to the 5th digit (by default)
    # //although can be overwritten by passing precision argument
    def print_system_coordinates(self, precision = 5):
        for planet in self.planets:
            print(np.around(planet.pos(0)[0], precision), np.around(planet.pos(0)[1], precision), np.around(planet.pos(0)[2], precision), end=' ')
        print(np.around(self.k_energy,precision), np.around(self.u_energy,precision), np.around(self.k_energy+self.u_energy,precision))
        print("")


    # Routine that implements the Euler integrator
    # //the number of iterations is set globally
    # //the routine is essentially a double for loop -> slow for large number of particles
    # //a routine for saving computational results on an output file is called for each iteration
    #
    # The Euler integrator is given by the following equations for position and velocities
    # //note that in practice this equations are vectorial
    # Eq(1) r(t+dt) = r(t) + v(t) * dt + 1/2m * dt^2 * f(t)
    # Eq(2) v(t+dt) = v(t) + dt/m * f(t)
    def evolve_euler(self):
        iter = 0
        while iter < ITERATIONS:

            # energies are computed at the beginning of each iterations
            self.compute_k_energy()
            self.compute_u_energy()

            # force computation at current coordinates (timestamp = 0)
            self.compute_system_force(0)
            for planet in self.planets:

                # remember that the sun (ID=0) is supposed stationary
                if(planet.ID != 0):
                    new_pos = planet.pos(0) + DT*planet.vel() + planet.net_force*DT*DT/planet.mass
                    new_vel = planet.vel() + DT*planet.net_force/planet.mass

                    planet.set_pos(new_pos, 1)
                    planet.set_vel(new_vel)

            iter += 1
            self.time += DT

            # after each iterations the coordinates need to be updated
            # and printed to standard output (thinned down by a factor of 100 (once every 50 days))
            self.update_coordinates()
            self.print_system_coordinates()


    # Routine that implements the Verlet integrator
    # //the number of iterations is set globally
    # //the routine is essentially a double for loop -> slow for large number of particles
    # //a routine for saving computational results on an output file is called for each iteration
    #
    # The Verlet integrator is given by the following equations for position and velocities
    # //note that in practice this equations are vectorial
    # Eq(3) r(t+dt) = 2r(t) -r(t-dt) + 1/2m * dt^2 * f(t)
    # Eq(4) v(t) = [r(t+dt) - r(t-dt)] / 2dt
    def evolve_verlet(self):
        iter = 0
        while iter<ITERATIONS:

            # energies are computed at the beginning of each iteration
            self.compute_k_energy()
            self.compute_u_energy()

            # force computation at current coordinates (timestamp = 0)
            self.compute_system_force(0)

            # If the current iteration is the first iteration it is necessary to
            # propagate the positions to the new "mid-positions", i.e shift the
            # initial conditions to first iteration. This is done with a simple Euler propagation of
            # the coordinates. Although the Euler scheme has larger errors a single iterations
            # has no influence on a 500 years simulation
            if(iter == 0):
                for planet in self.planets:
                    if(planet.ID!=0):
                        # simple Euler propagation of the positions
                        new_pos = planet.pos(0) + DT*planet.vel() + planet.net_force*DT*DT/planet.mass
                        planet.set_pos(new_pos, 1)

            else:
                for planet in self.planets:
                    if(planet.ID != 0):
                        new_pos = 2*planet.pos(0) - planet.pos(-1) + planet.net_force*DT*DT/planet.mass
                        new_vel = (new_pos - planet.pos(-1))/(2*DT)

                        planet.set_pos(new_pos,1)
                        planet.set_vel(new_vel)

            iter += 1
            self.time += DT

            # after each iterations the coordinates need to be updated
            # and printed to standard output
            self.update_coordinates()
            self.print_system_coordinates()


    # Routine that implements the Velocity-Verlet integrator
    # //the number of iterations is set globally
    # //the routine is slightly less efficient than Verlet and Euler since it needs an additional force evaluation for each step
    # //a routine for saving computational results on an output file is called for each iteration
    #
    # The Velocity-Verlet integrator is given by the following equations for position and velocities
    # //note that in practice this equations are vectorial
    # Eq(3) r(t+dt) = r(t) + v(t) * dt + 1/2m * dt^2 * f(t)
    # Eq(4) v(t+dt) = v(t) + dt/2m * [f(t) + f(t+dt)]
    def evolve_velocity_verlet(self):
        iter = 0
        while iter < ITERATIONS:

            # energies are computed at the beginning of each iteration
            self.compute_k_energy()
            self.compute_u_energy()

            # force computation at current coordinates (timestamp = 0)
            self.compute_system_force(0)
            for planet in self.planets:
                if(planet.ID != 0):
                    new_pos = planet.pos(0) + DT*planet.vel() + planet.net_force*DT*DT/planet.mass
                    planet.set_pos(new_pos,1)

                    # current force is saved for the velocity propagation
                    planet.v_verlet_net_force = planet.net_force

            # force computation at new coordinates (timestamp = 1)
            self.compute_system_force(1)
            for planet in self.planets:
                if(planet.ID != 0):
                    new_vel = planet.vel() + 0.5*DT*(planet.net_force+planet.v_verlet_net_force)/planet.mass

            iter += 1
            self.time += DT

            # after each iterations the coordinates need to be updated
            # and printed to standard output
            self.update_coordinates()
            self.print_system_coordinates()



if __name__ == '__main__':

    # Loading planets from file
    planets_name, planets_mass, planets_pos, planets_vel = load_planets()

    # Initializing Solar System class
    SolarSystem = System("SolarSystem")

    # Adding planets to solar system
    # //the sun is considered a planet
    # //the masses need the units of measure UNITS_OF_MASS = 10^29 Kg
    for i in range(len(planets_name)):
        SolarSystem.add_planet(i,planets_name[i], planets_mass[i]*UNITS_OF_MASS, planets_pos[i], planets_vel[i])

    # The system is made to evolve
    # //default timestep: half day
    # //default lenght of simulation: 500 years
    # //type of integrator is taken as argument
    if(len(sys.argv) == 1):
        print("Error: integrator not provided. Options (case sensitive) are euler, verlet, velocity-verlet")
        sys.exit()


    integrator = sys.argv[1]

    if(integrator == 'euler'):
        SolarSystem.evolve_euler()
    elif(integrator == 'verlet'):
        SolarSystem.evolve_verlet()
    elif(integrator == 'velocity-verlet'):
        SolarSystem.evolve_velocity_verlet()
    else:
        print("Error: undefined integrator. Options (case sensitive) are euler, verlet, velocity-verlet")
