#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 10: Modelling a water nano-droplet


*Comments*:
- The bond contribution of the force seems to work perfectly, and indeed oscillates around the
average value r_not (See plot1 in current directory)

- The angle contribution however seems too be strong (about an order of magnitude) but I was not able
to find the origin of such problem

- Probably it is due to the fact that I was not able to properly define the initial hydrogen positions with
a specific angle_not value. Somehow the rotation matrix I was using gave inconsistent results.

Latest update: July 12th 2021
"""

# Core namespaces
import numpy as np
from tqdm import tqdm
from numba import jit, njit, vectorize
from numba import types, typed, typeof, deferred_type

# Tailored modules/namespaces
from classmodule import Particle, Water_Molecule, Force, IntegratorNH




if __name__ == '__main__':


    m_oxygen = 2.6561E-26
    c_oxygen = -0.84*1.60217662E-19
    m_hydrogen = 1.6735E-27
    c_hydrogen = 0.42*1.60217662E-19

    force = Force(1058.,75.,1E-10,104.5,3.21E-10,1,1)
    integrator = IntegratorNH(0.5E-15, 300)


    """Initialize particles"""
    O = Particle(np.zeros(3), np.zeros(3), m_oxygen, 0, c_oxygen)
    H1 = Particle(np.zeros(3), np.zeros(3), m_hydrogen, 1, c_hydrogen)
    H2 = Particle(np.zeros(3), np.zeros(3), m_hydrogen, 1, c_hydrogen)

    """Initialize and populate water molecule"""
    water = Water_Molecule(0)
    water.load_atoms(O,H1,H2)
    water.distribute_atoms(force.rnot, force.anot)


    """Load molecule into force class"""
    molecules = typed.List() #specific Numba friendly lists
    molecules.append(water)
    force.add_molecules(molecules)


    """Test intramolecular force calculation"""
    force.initialize_forces()
    force.intra_molecular()


    """Load particle list into integrator class"""
    particles = typed.List() #specific Numba friendly lists
    particles.append(water.oxygen)
    particles.append(water.hydrogen1)
    particles.append(water.hydrogen2)
    integrator.add_particles(particles)




    """Test integration (multiple steps)"""
    bond_len = []
    angle = []
    for iter in tqdm(range(10000)):

        if iter%100 == 0 : bond_len.append(water.get_bond_len())
        if iter%100 == 0 : angle.append(water.get_angle())
        integrator.half_step()
        force.initialize_forces()
        force.intra_molecular()
        integrator.full_step()

    average_bond = np.mean(np.asarray(bond_len))
    average_angle = np.mean(np.asarray(angle))
    print(average_bond, force.rnot)
    print(average_angle, force.anot)


    #x = np.linspace(0, 10000*integrator.dt, num=len(bond_len))
    #printing.plot(False, x, bond_len, "Bond Length", "Time [s]", "Bond Length [m]", "Bond Length", "bond_len")
