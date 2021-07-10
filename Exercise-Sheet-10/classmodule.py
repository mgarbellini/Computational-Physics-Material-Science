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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" PHYSICAL CONSTANTS  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
KB = 1.38E-23
enot = 8.854E-12


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
    def __init__(self, id):

        self.ID = id
        self.angle = 0
        self.bond_len = 0

    def load_atoms(self, oxygen, hydrogen1, hydrogen2):
        self.hydrogen1 = hydrogen1
        self.hydrogen2 = hydrogen2
        self.oxygen = oxygen


    def molecule_from_seed(self, oxygen, hydrogen1, hydrogen2):
        self.hydrogen1 = hydrogen1
        self.hydrogen2 = hydrogen2
        self.oxygen = oxygen

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
    ('molecules', types.ListType(Water_Molecule.class_type.instance_type)),
    ('N', int32)
]
@jitclass(specforce)
class Force:

    def __init__(self, kbond, kangle, rnot, anot, sigma, epsilon, rcut):

        self.kbond = kbond
        self.kangle = kangle
        self.rnot = rnot
        self.anot = anot
        self.sigma = sigma
        self.rcut = rcut
        self.epsilon = epsilon
        self.N = 0

    def add_molecules(self, molecules):
        self.molecules = molecules
        self.N = len(self.molecules)

    def initialize_forces(self):
        for i in range(self.N):
            molecules[i].oxygen.force = np.zeros(3)
            molecules[i].hydrogen1.force = np.zeros(3)
            molecules[i].hydrogen2.force = np.zeros(3)

    def intra_molecular(self):
        _bond_force(self.molecules, self.N, self.kbond, self.rnot)
        _angle_force(self.molecules, self.N, self.kangle, self.anot)

    def inter_molecular(self):
        _inter_force(self.molecules, self.N, self.sigma, self.rcut, self.epsilon)


@njit()
def _bond_force(molecules, n_molecules, kbond, rnot):
    """Computes the bond contribution to the intra-molecular force"""
    for i in range(n_molecules):
        print("compute bond force")

@njit()
def _angle_force(molecules, n_molecules, kangle, anot):
    """Computes the angle contribution to the intra-molecular force"""
    for i in range(n_molecules):
        print("compute angle force")

@njit()
def _inter_force(molecules, n_molecules, sigma, rcut, e):
    """Computes the inter-molecular force between oxygen atoms, using a Lennard Jones
    12-6 potential and as well as a electrostatic Coulombic potential"""
    for i in range(n_molecules-1):
        for j in range(i+1, n_molecules):

            rx = molecules[i].oxygen.pos[0] - molecules[j].oxygen.pos[0]
            ry = molecules[i].oxygen.pos[1] - molecules[j].oxygen.pos[1]
            rz = molecules[i].oxygen.pos[2] - molecules[j].oxygen.pos[2]
            r = rx*rx + ry*ry + rz*rz
            print(r)

            if(r < rcut*rcut):
                fx = 48 * e * (sigma**12 * rx / r**7 - 0.5 * sigma**6 * rx / r**4)
                fy = 48 * e * (sigma**12 * ry / r**7 - 0.5 * sigma**6 * ry / r**4)
                fz = 48 * e * (sigma**12 * rz / r**7 - 0.5 * sigma**6 * rz / r**4)

                molecules[i].oxygen.force[0] += fx
                molecules[j].oxygen.force[0] -= fx

                molecules[i].oxygen.force[1] += fy
                molecules[j].oxygen.force[1] -= fy

                molecules[i].oxygen.force[2] += fz
                molecules[j].oxygen.force[2] -= fz



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""  CLASS: INTEGRATOR  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
specintegrator = [
    ('dt', float64),          # a simple scalar field
    ('n', int32),
    ('m', int32),
    ('T', float64),
    ('Q', float64),
    ('xi', float64),
    ('lns', float64),
    ('system_particles', types.ListType(Particle.class_type.instance_type)),          # an array field
]
@jitclass(specintegrator)
class IntegratorNH:

    def __init__(self, timestep, temp):

        self.dt = timestep
        self.n = 0
        self.m = 50
        self.T = temp
        self.Q = 0
        self.xi = 0
        self.lns = 0

    def add_particles(self, particles):
        self.system_particles = particles
        self.n = len(self.system_particles)
        self.Q = (3*self.n + 1) * KB * self.dt * self.T * self.m * self.m

    def half_step(self):
        for i in range(self.n):

            kinetic = 0
            # update half velocity
            self.system_particles[i].vel[0] = (self.system_particles[i].vel[0] + 0.5 * self.system_particles[i].force[0] * self.dt / self.system_particles[i].m) / (1 + self.xi * self.dt/2)
            self.system_particles[i].vel[1] = (self.system_particles[i].vel[1] + 0.5 * self.system_particles[i].force[1] * self.dt / self.system_particles[i].m) / (1 + self.xi * self.dt/2)
            self.system_particles[i].vel[2] = (self.system_particles[i].vel[2] + 0.5 * self.system_particles[i].force[2] * self.dt / self.system_particles[i].m) / (1 + self.xi * self.dt/2)


            # update position
            self.system_particles[i].pos[0] += self.system_particles[i].vel[0] * self.dt
            self.system_particles[i].pos[1] += self.system_particles[i].vel[1] * self.dt
            self.system_particles[i].pos[2] += self.system_particles[i].vel[2] * self.dt

            # kinetic energy
            kinetic += 0.5 * self.system_particles[i].m * (self.system_particles[i].vel[0]**2 + self.system_particles[i].vel[1]**2 + self.system_particles[i].vel[2]**2)

        # G factor
        G = (2*kinetic - 3*self.n*KB*self.T)/self.Q
        self.lns += self.xi * self.dt + 0.5 * G * self.dt * self.dt
        self.xi += G*self.dt


    def full_step(self):
        for i in range(self.n):

            # updates velocities to full step
            self.system_particles[i].vel[0] += 0.5*self.dt*(self.system_particles[i].force[0]/self.system_particles[i].m - self.xi*self.system_particles[i].vel[0])
            self.system_particles[i].vel[1] += 0.5*self.dt*(self.system_particles[i].force[1]/self.system_particles[i].m - self.xi*self.system_particles[i].vel[1])
            self.system_particles[i].vel[2] += 0.5*self.dt*(self.system_particles[i].force[2]/self.system_particles[i].m - self.xi*self.system_particles[i].vel[2])
