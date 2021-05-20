import settings
import numpy as np
import math
import sys

def WriteEnergy(fileenergy, itime):
    
    fileenergy.write("%i %e %e %e %e %e \n" % (itime, settings.Energy.ep, settings.Energy.ek, settings.Energy.ekx, settings.Energy.eky, settings.Energy.ekz))

def WriteTrajectory(fileoutput, itime):

    fileoutput.write("ITEM: TIMESTEP \n")
    fileoutput.write("%i \n" % itime)
    fileoutput.write("ITEM: NUMBER OF ATOMS \n")
    fileoutput.write("%i \n" % settings.nparticles)
    fileoutput.write("ITEM: BOX BOUNDS \n")
    fileoutput.write("%e %e xlo xhi \n" % (settings.box.xlo*1e10, settings.box.xhi*1e10))
    fileoutput.write("%e %e xlo xhi \n" % (settings.box.ylo*1e10, settings.box.yhi*1e10))
    fileoutput.write("%e %e xlo xhi \n" % (settings.box.zlo*1e10, settings.box.zhi*1e10))
    fileoutput.write("ITEM: ATOMS id type x y z \n")
    
    for i in range(0, settings.nparticles):
        x = settings.xi[i] % settings.box.lx
        y = settings.yi[i] % settings.box.ly
        z = settings.zi[i] % settings.box.lz
        fileoutput.write("%i %i %e %e %e \n" % (i, i, x*1e10, y*1e10, z*1e10))
        


    
    