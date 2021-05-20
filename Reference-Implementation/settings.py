#settings

# velocity: m*s^-1
# position: m
# acceleration: ms^-2
# energy: joule
# temperature: K

import numpy as np

def init():
    global nparticles        # total number of particles
    global nsteps            # number of time step to analyze
    nsteps = 2000
    global mass              # mass of the LJ particles (kg)
    mass = 105.5206e-27
    global kb                # boltzmann's constant (m^2*kg*s^-2*K^-1)
    kb =1.3806503e-23
    global Tdesired          # temperature of the experiment in K
    Tdesired = 300.
    global eps               # eps in LJ (kg*m^2*s^-2)
    eps = 0.5*kb*Tdesired
    global r0                # r0 in LJ (m)
    r0 = 2.55e-10
    global cutoff            # cutoff arbitrary at 15 angstrom
    cutoff = 2.5*r0
    global deltat            # time step (s)
    deltat = 1.e-15
    global p                  # list for storing all the particle
    p = []



    global particle

    class particle:

        def __init__(self, x, y, z, vx, vy, vz, fx, fy, fz):
            self.x = x
            self.y = y
            self.z = z
            self.vx = vx
            self.vy = vy
            self.vz = vz
            self.fx = fx
            self.fy = fy
            self.fz = fz

    global box

    class box:

        def __init__(self, xlo, xhi, ylo, yhi, zlo, zhi, lx, ly, lz):
            self.xlo = xlo
            self.xhi = xhi
            self.ylo = ylo
            self.yhi = yhi
            self.zlo = zlo
            self.zhi = zhi

    # parameters to build the initial configuration (atoms on lattice)
    global deltaxyz          # lattice used to build the initial configuration
#    deltaxyz = 5.10e-10
#    deltaxyz = 5.10e-10
    global n1
    n1 = 8
    global n2
    n2 = 8
    global n3
    n3 = 8
    global ntot
    ntot = n1*n2*n3

    # initialize box dimension in angstrom
    box.xlo = 0.
#    box.xhi = n1*deltaxyz
    box.xhi = 17.2*r0
    box.ylo = 0.
#    box.yhi = n2*deltaxyz
    box.yhi = 17.2*r0
    box.zlo = 0.
#    box.zhi = n3*deltaxyz
    box.zhi = 17.2*r0
    box.lx = box.xhi - box.xlo
    box.ly = box.yhi - box.ylo
    box.lz = box.zhi - box.zlo

    deltaxyz = box.lx / n1  #5.10e-10
    print("spacing of the lattice" , deltaxyz)

    global Energy    # class for energy (potential, kinetic)

    class Energy:
        def __init__(self, ep, ek, ekx, eky, ekz):
            self.ep = ep
            self.ek = ek
            self.ekx = ekx
            self.eky = eky
            self.ekz = ekz

    global debug
    debug = 0  # 1 == debug; 0 == no debug
    global Trescale
    Trescale = 1  # 1 == rescale velocity at desired temperature every N steps; 0 == no rescaling


# for vectorization
    global xi
    xi = np.zeros(shape=(n1*n2*n3))
    global yi
    yi = np.zeros(shape=(n1*n2*n3))
    global zi
    zi = np.zeros(shape=(n1*n2*n3))
    global vix
    vix = np.zeros(shape=(n1*n2*n3))
    global viy
    viy = np.zeros(shape=(n1*n2*n3))
    global viz
    viz = np.zeros(shape=(n1*n2*n3))
    global sumfix
    sumfix = np.zeros(shape=(n1*n2*n3))
    global sumfiy
    sumfiy = np.zeros(shape=(n1*n2*n3))
    global sumfiz
    sumfiz = np.zeros(shape=(n1*n2*n3))
