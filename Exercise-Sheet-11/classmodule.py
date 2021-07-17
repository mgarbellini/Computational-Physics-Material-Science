#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* CLASS MODULE *
Contains the core classes needed for the Water.py main file:
- Particle
- Molecule (inherits the particle class)
- Force (inherits the molecule class)
- IntegratorNH (inherits the particle class)

Latest update: July 12th 2021
"""

import numpy as np
from numba import njit
from numba import int32, float64
from numba import types, typed, typeof, deferred_type
from numba.experimental import jitclass
from numba import prange

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

    def norm(self):
        return np.linalg.norm(self.pos)



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
        """Loads the molecule with the input atoms. Atoms must be
        altready fully defined (no further manipulation needed)"""
        self.hydrogen1 = hydrogen1
        self.hydrogen2 = hydrogen2
        self.oxygen = oxygen

    def distribute_atoms(self, rzero, azero):
        """Distributes the hydrogen atoms spatially with respect
        to the central oxygen atom"""

        #set same position
        self.hydrogen1.pos[0] = np.random.uniform(-1.,1.)
        self.hydrogen1.pos[1] = np.random.uniform(-1.,1.)
        self.hydrogen1.pos[2] = np.random.uniform(-1.,1.)

        #rotate second hydrogen position with respect to oxygen of azero degrees
        sign = np.random.choice(np.array([-1,1]))
        c = np.cos(sign*azero)
        s = np.sin(sign*azero)

        Rx = np.array(((1,0,0),(0,c,-s),(0,s,c)), dtype = float64)
        Ry = np.array(((c,0,s),(0,1,0),(-s,0,c)), dtype = float64)
        Rz = np.array(((c,-s,0),(s,c,0),(0,0,1)), dtype = float64)
        axis = np.random.choice(np.array((0,1,2)))


        if axis == 0: self.hydrogen2.pos = Rx @ self.hydrogen1.pos
        elif axis == 1: self.hydrogen2.pos = Ry @ self.hydrogen1.pos
        else: self.hydrogen2.pos = Rz @ self.hydrogen1.pos

        #normalize positions so that the bond lenght is correct
        self.hydrogen1.pos *= rzero/np.linalg.norm(self.hydrogen1.pos)
        self.hydrogen2.pos *= rzero/np.linalg.norm(self.hydrogen2.pos)



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


    def get_distance_OH(self, id):

        if id == 1: dist = self.hydrogen1.pos - self.oxygen.pos
        else: dist = self.hydrogen2.pos - self.oxygen.pos

        return dist


    def get_angle(self):

        v1 = self.hydrogen1.pos/np.linalg.norm(self.hydrogen1.pos)
        v2 = self.hydrogen2.pos/np.linalg.norm(self.hydrogen2.pos)

        dot = np.dot(v1, v2)

        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot =  -1.0
        else:
            dot = dot

        self.angle = np.arccos(dot)
        return self.angle



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: POLYMER CHAIN  """""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
specpolymerchain = [
    ('ID', int32),
    ('atoms', types.ListType(Particle.class_type.instance_type)),
    ('bond', float64),
    ('N', int32)
]
@jitclass(specpolymerchain)
class Polymer_Chain:
    """Polymer-Chain class containing N particles"""
    def __init__(self, id, bond_len):

        self.ID = id
        self.bond = bond_len
        self.N = 0

    def load_particles(self, particles):
        """Loads the particles and form the chain. Atoms must be
        altready fully defined (no further manipulation needed)"""
        self.atoms = particles
        self.N = len(self.atoms)

    def distribute_particles(self):
        """Distributes the particles as a straight chain with a given bond length,
        and assigns the LJ_indices"""
        for i in range(self.N):
            self.atoms[i].pos[0] = i*self.bond
            self.atoms[i].pos[1] = 0
            self.atoms[i].pos[2] = 0

    def reset_chain(self):
        for i in range(self.N):
            self.atoms[i].vel = np.zeros(3)
            self.atoms[i].force = np.zeros(3)
            self.atoms[i].pos[0] = i*self.bond
            self.atoms[i].pos[1] = 0
            self.atoms[i].pos[2] = 0

    def get_bond_len(self, i, j):
        """Computes the bond length between particle i and particle j. If i and j
        are not consecutives particles get_bond_len returns the distance and not the bond len"""
        rijx = self.atoms[i].pos[0] - self.atoms[j].pos[0]
        rijy = self.atoms[i].pos[1] - self.atoms[j].pos[1]
        rijz = self.atoms[i].pos[2] - self.atoms[j].pos[2]
        rij = np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz)

        return rij/2

    def end_distance(self):
        r = -self.atoms[0].pos + self.atoms[self.N-1].pos
        return np.linalg.norm(r)

    def end_to_end_distance(self):
        distance = _end_to_end_distance(self.atoms, self.N)
        return distance

@njit()
def _end_to_end_distance(atoms, N):
    sum = 0
    for i in prange(N):
        for j in prange(N):
            sum += atoms[i].pos[0]*atoms[j].pos[0] + atoms[i].pos[1]*atoms[j].pos[1] + atoms[i].pos[2]*atoms[j].pos[2]
    return sum

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: FORCES """""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
specforce = [
    ('kbond', float64),
    ('rnot', float64),
    ('sigma', float64),
    ('epsilon', float64),
    ('polymer_chain', Polymer_Chain.class_type.instance_type),
]
@jitclass(specforce)
class Force:

    def __init__(self, kbond, rnot, sigma, epsilon):

        self.kbond = kbond
        self.rnot = rnot
        self.sigma = sigma
        self.epsilon = epsilon



    """""""POLYMER CHAIN FORCE"""""""
    def add_polymer_chain(self, chain):
        self.polymer_chain = chain
        for i in range(self.polymer_chain.N):
            self.polymer_chain.atoms[i].force = np.zeros(3)

    def initialize_force_polymer(self):
        for i in range(self.polymer_chain.N):
            self.polymer_chain.atoms[i].force = np.zeros(3)

    def force_polymer_chain(self):
        _bond_force_polymer(self.polymer_chain, self.rnot, self.kbond)
        _lennard_jones_polymer(self.polymer_chain, self.sigma, self.epsilon)


@njit(parallel = False)
def _bond_force_polymer(chain, rnot, kbond):
    """Computes the bond force between consecutives atoms in chain"""
    for i in prange(chain.N - 1):
        r = chain.atoms[i+1].pos - chain.atoms[i].pos
        dist = np.linalg.norm(r)
        f = kbond*(dist - rnot)*r/dist
        chain.atoms[i].force += f
        chain.atoms[i+1].force -= f


@njit(parallel = False)
def _lennard_jones_polymer(chain, sigma, e):
    """Computes the LJ force between non consecutive atoms in chain"""
    for i in prange(chain.N-2):
        rij = chain.atoms[i].pos - chain.atoms[i+2].pos
        r = rij[0]**2 + rij[1]**2 + rij[2]**2

        fx = 48 * e * (sigma**12 * rij[0] / r**7 - 0.5 * sigma**6 * rij[0] / r**4)
        fy = 48 * e * (sigma**12 * rij[1] / r**7 - 0.5 * sigma**6 * rij[1] / r**4)
        fz = 48 * e * (sigma**12 * rij[2] / r**7 - 0.5 * sigma**6 * rij[2] / r**4)

        chain.atoms[i].force[0] += fx
        chain.atoms[i+2].force[0] -= fx

        chain.atoms[i].force[1] += fy
        chain.atoms[i+2].force[1] -= fy

        chain.atoms[i].force[2] += fz
        chain.atoms[i+2].force[2] -= fz



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
