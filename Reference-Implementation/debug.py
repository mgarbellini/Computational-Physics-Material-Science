#!/usr/bin/env python2
import settings

def WriteAtoms(step):
    
    fileatoms = open("atoms_trajectory.txt","a")
    fileatoms.write("ITEM: TIMESTEP \n")
    fileatoms.write("%i \n" % step)
    fileatoms.write("ITEM: NUMBER OF ATOMS \n")
    fileatoms.write( "%i \n" % settings.nparticles)
    fileatoms.write("ITEM: BOX BOUNDS pp pp pp \n")
    fileatoms.write("%e %e \n" % (settings.box.xlo*1e10, settings.box.xhi*1e10))
    fileatoms.write("%e %e \n" % (settings.box.ylo*1e10, settings.box.yhi*1e10))
    fileatoms.write("%e %e \n" % (settings.box.zlo*1e10, settings.box.zhi*1e10))
    fileatoms.write("ITEM: ATOMS id x y z \n")
    for i in range(0, settings.nparticles):
        fileatoms.write("%i %e %e %e \n" % (i, settings.p[i].x*1e10, settings.p[i].y*1e10, settings.p[i].z*1e10))
    fileatoms.close()
    
def WriteVelocity(step):
    
    fileatoms = open("atoms_velocity.txt","w")
    fileatoms.write("ITEM: TIMESTEP \n")
    fileatoms.write("%i \n" % step)
    fileatoms.write("ITEM: NUMBER OF ATOMS \n")
    fileatoms.write( "%i \n" % settings.nparticles)
    fileatoms.write("ITEM: BOX BOUNDS pp pp pp \n")
    fileatoms.write("%e %e \n" % (settings.box.xlo, settings.box.xhi))
    fileatoms.write("%e %e \n" % (settings.box.ylo, settings.box.yhi))
    fileatoms.write("%e %e \n" % (settings.box.zlo, settings.box.zhi))
    fileatoms.write("ITEM: ATOMS id x y z \n")
    for i in range(0, settings.nparticles):
        fileatoms.write("%i %e %e %e %e %e %e \n" % (i, settings.p[i].x, settings.p[i].y, settings.p[i].z, settings.p[i].vx, settings.p[i].vy, settings.p[i].vz))
    fileatoms.close()

def WriteForces(step):
    
    fileatoms = open("atoms_forces.txt","w")
    fileatoms.write("ITEM: TIMESTEP \n")
    fileatoms.write("%i \n" % step)
    fileatoms.write("ITEM: NUMBER OF ATOMS \n")
    fileatoms.write( "%i \n" % settings.nparticles)
    fileatoms.write("ITEM: BOX BOUNDS pp pp pp \n")
    fileatoms.write("%e %e \n" % (settings.box.xlo, settings.box.xhi))
    fileatoms.write("%e %e \n" % (settings.box.ylo, settings.box.yhi))
    fileatoms.write("%e %e \n" % (settings.box.zlo, settings.box.zhi))
    fileatoms.write("ITEM: ATOMS id x y z \n")
    for i in range(0, settings.nparticles):
        fileatoms.write("%i %e %e %e %e %e %e \n" % (i, settings.p[i].x, settings.p[i].y, settings.p[i].z, settings.p[i].fx, settings.p[i].fy, settings.p[i].fz))
    fileatoms.close()