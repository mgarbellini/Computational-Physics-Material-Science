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

    with open("planets.dat") as file_in:
        planet_name = ['Sun',]
        planet_pos = [[0,0,0],]
        planet_vel = [[0,0,0],]
        for line in file_in:
            sl = line.split()
            planet_name.append(sl[6].replace("!%", ""))
            planet_pos.append([float(sl[0]), float(sl[1]), float(sl[2])])
            planet_vel.append([float(sl[3]), float(sl[4]), float(sl[5])])

    with open("mass.dat") as file_in:
        planet_mass = []
        for line in file_in:
            planet_mass.append(float(line.split()[0]))

    return planet_name, planet_mass, planet_pos, planet_vel


class Object:

    def __init__(self, name, mass, init_position, init_velocity):

        self.name = name
        self.mass = mass*UNITS_OF_MASS
        self.init_position = np.array(init_position, dtype = np.float)*ASTRO_POS
        self.init_velocity = np.array(init_velocity, dtype = np.float)*ASTRO_POS/ASTRO_DAY
        self.pos = [self.init_position,]
        self.vel = [self.init_velocity,]
        self.net_force = 0


class System:

    def __init__(self, sun):

        self.sun = sun
        self.planets = []
        self.t = 0
        self.t_id = 0
        self.n_planets = 0
        self.total_energy = None
        self.potential_energy = None
        self.kinetic_energy = None


    def add_planet(self, planet):

        self.planets.append(planet)
        self.n_planets += 1

    def compute_fmatrix(self, t_id):

        f_matrix = np.zeros((3,self.n_planets, self.n_planets), dtype = np.float)
        iterable = [0,1,2,3,4,5,6,7,8]
        for dir in range(3):
            for p1, p2 in itertools.combinations_with_replacement(iterable,2):
                if(p1!=p2):
                    f_matrix[dir,p1,p2] = (GRAVITATIONAL_CONST*self.planets[p1].mass*self.planets[p2].mass)/(self.planets[p1].pos[t_id][dir] - self.planets[p2].pos[t_id][dir])**2
                    f_matrix[dir,p2,p1] = f_matrix[dir,p1,p2]
                else:
                    f_matrix[dir,p1,p2] = (GRAVITATIONAL_CONST*self.sun.mass*self.planets[p1].mass)/((self.planets[p1].pos[t_id][dir])**2)

        return np.sum(f_matrix, axis = 1)

    def compute_system_energy(self):
        iterable = [0,1,2,3,4,5,6,7,8]

        p_matrix = np.zeros((self.n_planets, self.n_planets),dtype = np.float)
        k_matrix = np.zeros((self.n_planets, 1), dtype = np.float)

        for p1, p2 in itertools.combinations_with_replacement(iterable,2):
            if(p1!=p2):
                p_matrix[p1,p2] = (-GRAVITATIONAL_CONST*self.planets[p1].mass*self.planets[p2].mass)/(np.sqrt((np.linalg.norm(self.planets[p1].pos[self.t_id] - self.planets[p2].pos[self.t_id]))))
            else:
                p_matrix[p1,p2] = (-GRAVITATIONAL_CONST*self.sun.mass*self.planets[p1].mass)/(np.sqrt(np.linalg.norm(self.planets[p1].pos[self.t_id])))
                k_matrix[p1,0] = 0.5*self.planets[p1].mass*np.linalg.norm(self.planets[p1].vel[self.t_id])

        self.potential_energy = np.sum(p_matrix)
        self.kinetic_energy = np.sum(k_matrix)
        self.total_energy = np.sum(p_matrix) + np.sum(k_matrix)





    def evolve_euler(self, dt):

        net_forces_matrix = self.compute_fmatrix(self.t_id)
        for planet_id, planet  in enumerate(self.planets):
            position = planet.pos[self.t_id] + dt*planet.vel[self.t_id] + 0.5/planet.mass*dt*dt*net_forces_matrix[:,planet_id]
            velocity = planet.vel[self.t_id] + dt/planet.mass*net_forces_matrix[:,planet_id]
            planet.pos.append(position)
            planet.vel.append(velocity)

        self.t_id += 1


if __name__ == '__main__':

    planets_name, planets_mass, planets_pos, planets_vel = load_planets()

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
