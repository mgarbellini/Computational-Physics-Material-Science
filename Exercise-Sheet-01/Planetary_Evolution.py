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


GRAVITATIONAL_CONST = 1
#UNITS_OF_MASS = 10E29 """units of mass in kg"""

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
        self.mass = mass
        self.init_position = np.array(init_position, dtype = np.float)
        self.init_velocity = np.array(init_velocity, dtype = np.float)
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


    def add_planet(self, planet):

        self.planets.append(planet)
        self.n_planets += 1

    def compute_net_forces(self, t_id):

        f_matrix = np.zeros((3,self.n_planets, self.n_planets), dtype = np.float)
        iterable = [0,1,2,3,4,5,6,7,8]
        for dir in range(3):
            for p1, p2 in itertools.combinations_with_replacement(iterable,2):
                if(p1!=p2):
                    f_matrix[dir,p1,p2] = GRAVITATIONAL_CONST*self.planets[p1].mass*self.planets[p2].mass/(self.planets[p1].pos[t_id][dir] - self.planets[p2].pos[t_id][dir])
                    f_matrix[dir,p1,p2] = f_matrix[dir,p2,p1]
                else:
                    f_matrix[dir,p1,p2] = GRAVITATIONAL_CONST*self.sun.mass*self.planets[p1].mass/self.planets[p1].pos[t_id][dir]
        return f_matrix

    def evolve_euler(self, dt):

        net_forces_matrix = self.compute_net_forces(self.t_id)
        for planet, planet_id in enumerate(self.planets):
            position = planet.pos[t_id] + dt*planet.vel[t_id] + 0.5/planet.mass*dt*dt*net_forces_matrix[:,planet_id]
            velocity = planet.vel[t_id] + dt/planet.mass*net_forces_matrix[:,planet_id]
            planet.pos.append(position)
            planet.vel.append(velocity)

        self.t_id += 1


if __name__ == '__main__':

    planets_name, planets_mass, planets_pos, planets_vel = load_planets()

    Sun = Object(planets_name[0], planets_mass[0], planets_pos[0], planets_vel[0])
    SolarSystem = System(Sun)

    for i in range(1, len(planets_name)):
        SolarSystem.add_planet(Object(planets_name[i], planets_mass[i], planets_pos[i], planets_vel[i]))

    for planet in SolarSystem.planets:
        print(planet.name)

    print(SolarSystem.compute_net_forces(0))
