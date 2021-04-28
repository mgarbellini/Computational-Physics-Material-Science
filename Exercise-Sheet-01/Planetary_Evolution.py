"""
M. Garbellini
matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 01
"""


import numpy as np


class Object:
    def __init__(self, name, mass, init_position, init_velocity):
        self.name = name
        self.mass = mass
        self.init_position = init_position
        self.init_velocity = init_velocity
        self.pos = [self.init_position,]
        self.vel = [self.init_velocity,]

class System:
    def __init__(self, sun):
        self.sun = sun
        self.planets = []
        self.timestamp = 0

    def add_planet(self, planet):
        self.planet.append(planet)

    def evolve(self, timestep, integrator)
        for planet in self.planets :
            if (integrator == "E"):
                run_euler_integrator()
            elif (integrator == "V"):
                run_verlet_integrator()
            elif (integrator == "VV")
                run_velocity_verlet_integrator()
            else:
                print("Error: undefined integrator type.
                        \n Please enter E - Euler,
                        V - Verlet, VV - Velocity Verlet")
