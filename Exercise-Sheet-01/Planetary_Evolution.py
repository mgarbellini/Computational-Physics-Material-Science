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
TIMESTEP = 43200

def load_planets():

    position = np.loadtxt("planets.txt", usecols=(0,1,2))
    velocity = np.loadtxt("planets.txt", usecols=(3,4,5))
    names = np.loadtxt("planets.txt", dtype=object, usecols=(6) )
    mass = np.loadtxt("mass.txt", usecols=(0))

    print(names[0])
    return names, mass, position, velocity


class System:

    def __init__(self, name, mass, pos, vel):

        self.planet_name = name
        self.t = 0
        self.n_planets = len(planet_name)

        self.p_mass = np.resize(mass, (len(mass), len(mass)))*UNITS_OF_MASS
        self.p_mass_matrix = np.fill_diagonal(np.multiply(self.p_mass, np.transpose(self.p_mass)),0)

        self.p_pos_previous = None
        self.p_vel_previous = None

        self.p_pos_current = None
        self.p_vel_current = None

        self.p_pos_next = None
        self.p_vel_next = None


        self.total_energy = None
        self.potential_energy = None
        self.kinetic_energy = None

    def set_initial_condition(self, position, velocity):
        if(self.t != 0):
            print("Error: setting intial condition for t>0")
            dim = len(position[0,:])
            self.p_pos_current = np.stack(( np.resize(position[0,:], (dim, dim)),
                                            np.resize(position[1,:], (dim, dim)),
                                            np.resize(position[2,:], (dim, dim))),
                                            axis=0)
            self.p_vel_current = velocity


    def net_force(self, pos):
        distance_square_matrix = (np.abs(pos - pos.transpose((0, 2, 1))))**2
        net_force_matrix = GRAVITATIONAL_CONST*np.multiply(np.reciprocal(distance_square_matrix),np.stack(( self.p_mass_matrix,
                                                                                                                self.p_mass_matrix,
                                                                                                                self.p_mass_matrix),
                                                                                                                axis=0))
        return sum(net_force_matrix, axis=1)

    #def k_energy(self, vel):

    #def u_energy(self, pos):


    def update_current(self):

        self.p_pos_previous = self.p_pos_current
        self.p_vel_previous = self.p_vel_current

        self.p_pos_current = self.p_pos_next
        self.p_vel_current = self.p_vel_next

        self.p_pos_next = None
        self.p_vel_next = None

    #def evolve_euler(self, iterations):

    #    while iter < iterations:

"""
    def evolve_euler(self, dt):

        net_forces_matrix = self.compute_fmatrix(self.t_id)
        for planet_id, planet  in enumerate(self.planets):
            position = planet.pos[self.t_id] + dt*planet.vel[self.t_id] + 0.5/planet.mass*dt*dt*net_forces_matrix[:,planet_id]
            velocity = planet.vel[self.t_id] + dt/planet.mass*net_forces_matrix[:,planet_id]
            planet.pos.append(position)
            planet.vel.append(velocity)

        self.t_id += 1
"""

if __name__ == '__main__':

    planets_name, planets_mass, planets_pos, planets_vel = load_planets()
    """
    Sun = Object(planets_name[0], planets_mass[0], planets_pos[0], planets_vel[0])
    SolarSystem = System(Sun)

    for i in range(1, len(planets_name)):
        p = Object(planets_name[i], planets_mass[i], planets_pos[i], planets_vel[i])
        SolarSystem.add_planet(p)

    time_stamp = time.perf_counter()
    for i in range(10):
        print(i)
        SolarSystem.compute_system_energy()
        SolarSystem.evolve_euler(TIMESTEP)
    print(time.perf_counter() - time_stamp)


    with plt.style.context(['science', 'dark_background']):
        fig, ax = plt.subplots()
        for planet in SolarSystem.planets:
            ax.plot([a[0] for a in planet.pos], [a[1] for a in planet.pos], label=planet.name)
            ax.legend(title='Planets orbits')
            ax.autoscale(tight=True)
        fig.savefig('./fig1.pdf')
    """
