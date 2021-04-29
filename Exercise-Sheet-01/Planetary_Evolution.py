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


import numpy as np
import itertool


G = 128 """gravitational constant"""
M = 25 """mass of sun"""


class Object:

    def __init__(self, name, mass, init_position, init_velocity):

        self.name = name
        self.mass = mass
        self.init_position = init_position
        self.init_velocity = init_velocity
        self.pos = [self.init_position,]
        self.vel = [self.init_velocity,]
        self.net_force = 0

    @classmethod
    def load_from_file(cls, filename):
        instances = []
        



class System:

    def __init__(self, sun):

        self.sun = sun
        self.planets = []
        self.t = 0
        self.t_id = 0
        self.n_planets = 0


    def add_planet(self, planet):

        self.planet.append(planet)
        self.n_planets += 1

    def compute_net_forces(self, t_id):

        f_matrix = np.zeros((3,self.n_planets, self.n_planets), dtype = np.float)
        for dir in range(3):
            for p1, p2 in itertool.combinations_with_replacement(np.arange(0,self.n_planets), 1, dtype = int64),2):
                if(p1!=p2):
                    f_matrix[dir,p1,p2] = G*self.planets[p1].mass*self.planets[p2].mass/(self.planets[p1].pos[t_id][dir] - self.planets[p2].pos[t_id][dir])
                    f_matrix[dir,p1,p2] = f_matrix[dir,p2,p1]
                else:
                    f_matrix[dir,p1,p2] = G*self.sun.mass*self.planets[i].mass/self.planets[i].pos[t_id][dir]
        return sum(f_matrix, axis = 1)


    def evolve_euler(self, dt):

        net_forces_matrix = self.compute_net_forces(self.t_id)
        for planet, planet_id in enumerate(self.planets):
            position = planet.pos[t_id] + dt*planet.vel[t_id] + 0.5/planet.mass*dt*dt*net_forces_matrix[:,planet_id]
            velocity = planet.vel[t_id] + dt/planet.mass*net_forces_matrix[:,planet_id]
            planet.pos.append(position)
            planet.vel.append(velocity)

        self.t_id += 1


if __name__ == '__main__':
