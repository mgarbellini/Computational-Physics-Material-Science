#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* CLASS MODULE *
Contains the particles and molecules class implementations

Latest update: July 9th 2021
"""

import numpy as np
from numba import njit
from numba import int32, float64    # import the types
from numba import types, typed, typeof, deferred_type
from numba.experimental import jitclass


KB = 666

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: PARTICLE """""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
specparticle = [
    ('ID', int32),
    ('mID', int32),
    ('type', int32),
    ('pos', float64[:]),
    ('vel', float64[:]),
    ('force', float64[:]),
    ('m', float64),
    ('charge', float64),
]
@jitclass(specparticle)
class Particle:
    """Particle class containing positions, velocities, mass, particle-type"""
    def __init__(self,position, velocity, mass, type, charge = 0, id = 0):

        self.ID = id
        self.mID = 0 #molecule ID (if particle is put into molecule)
        self.type = type
        self.pos = position
        self.vel = velocity
        self.force = np.zeros(3)+1
        self.m = mass
        self.charge = charge

    def kinetic_energy(self):
        kinetic = self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2
        return kinetic




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: WATER MOLECULE """""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

specmolecule = [
    ('ID', int32),
    ('oxygen', Particle.class_type.instance_type),
    ('hydrogen1', Particle.class_type.instance_type),
    ('hydrogen2', Particle.class_type.instance_type),
    ('angle', float64),
    ('bond_len', float64),
]
@jitclass(specmolecule)
class Water_Molecule:
    """Water Molecule class containing two hydrogen and one oxygen"""
    def __init__(self, id, oxygen, hydrogen1, hydrogen2):

        self.ID = id
        self.hydrogen1 = hydrogen1
        self.hydrogen2 = hydrogen2
        self.oxygen = oxygen
        self.angle = 0
        self.bond_len = 0


    def get_bond_len(self):

        rijx = self.oxygen.pos[0] - self.hydrogen1.pos[0]
        rijy = self.oxygen.pos[1] - self.hydrogen1.pos[1]
        rijz = self.oxygen.pos[2] - self.hydrogen1.pos[2]
        rij = np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz)

        rijx = self.oxygen.pos[0] - self.hydrogen2.pos[0]
        rijy = self.oxygen.pos[1] - self.hydrogen2.pos[1]
        rijz = self.oxygen.pos[2] - self.hydrogen2.pos[2]
        rij += np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz)

        return rij/2


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: FORCES """""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
specforce = [
    ('kbond', float64),
    ('kangle', float64),
    ('rnot', float64),
    ('anot', float64),
    ('sigma', float64),
    ('rcut', float64),
    ('epsilon', float64),
    #('molecules', Water_Molecule.class_type.instance_type),
    ('molecules', types.ListType(Water_Molecule.class_type.instance_type)),
    ('N', int32)
]
@jitclass(specforce)
class Force:

    def __init__(self, kbond, kangle, rnot, anot, sigma, rcut):

        self.kbond = kbond
        self.kangle = kangle
        self.rnot = rnot
        self.anot = anot
        self.sigma = sigma
        self.rcut = rcut
        self.N = 0

    def add_molecules(self, molecules):
        self.molecules = molecules
        #self.N = len(self.molecules)





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""  CLASS: INTEGRATOR  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def _half_step(p, v, f, m, xi, dt):
    """Integrates the system given the NH-Thermostat velocity-verlet scheme"""

    # updates half step velocities
    v[0] = (v[0] + 0.5 * f[0] * dt / m) / (1 + xi * dt/2)
    v[1] = (v[1] + 0.5 * f[1] * dt / m) / (1 + xi * dt/2)
    v[2] = (v[2] + 0.5 * f[2] * dt / m) / (1 + xi * dt/2)

    # updates half step positions
    p[0] += v[0] * dt
    p[1] += v[1] * dt
    p[2] += v[2] * dt

    # computes kinetic energy of particle
    k = 0.5 * m * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    return p, v, k

def _full_step(v, f, m, xi, dt):

    # updates velocities to full step
    v[0] = v[0] + 0.5*dt*(f[0]/m - xi*v[0])
    v[1] = v[1] + 0.5*dt*(f[1]/m - xi*v[1])
    v[2] = v[2] + 0.5*dt*(f[2]/m - xi*v[2])

    return v


spec = [
    ('timestep', float64),          # a simple scalar field
    ('n', int32),
    ('m', int32),
    ('T', int32),
    ('Q', float64),
    ('xi', float64),
    ('lns', float64),
    ('array', float64[:]),          # an array field
]
@jitclass(spec)
class IntegratorNH:

    def __init__(self, timestep, temp, particles = None):

        self.dt = timestep
        self.n = 0
        self.m = 50
        self.T = temp
        self.Q = 0
        self.xi = 0
        self.lns = 0
        self.system_particles = particles #should be a list of particles

    def add_particles(self, particles):
        self.system_particles = particles
        self.n = len(particles)
        self.Q = (3*self.n + 1) * KB * self.dt * self.T * self.m * self.m

    def half_step(self):

        kinetic = 0
        for particle in self.system_particles:
            particle.pos, particle.vel, k = _half_step(particle.pos, particle.vel, particle.force, particle.m,
                                                    self.xi, self.dt)

            kinetic += k

        G = (2*kinetic - 3*self.n*KB*self.T)/self.Q

        self.lns += self.xi * self.dt + 0.5 * G * self.dt * self.dt
        self.xi += G*self.dt

    def full_step(self):

        for particle in self.system_particles:
            particle.vel = _full_step(particle.vel, particle.force, particle.m, self.xi, self.dt)
