import settings
import random
import math
import operator
import debug
import numpy as np


def InitializeAtoms():
    
    print("xlo = %e Angstrom and yhi = %e Angstrom" % (settings.box.xlo, settings.box.xhi))   
    
    
    nx = 0
    ny = 0
    nz = 0
    n = 0
    while nx < settings.n1:
        ny = 0
        while ny < settings.n2:
            nz = 0
            while nz < settings.n3:
                x = nx * settings.deltaxyz
                y = ny * settings.deltaxyz
                z = nz * settings.deltaxyz
                
                vx = 0.5 - random.randint(0, 1)
                vy = 0.5 - random.randint(0, 1)
                vz = 0.5 - random.randint(0, 1)
                #vz = 0.
                
                
                settings.xi[n] = x
                settings.yi[n] = y
                settings.zi[n] = z
                settings.vix[n] = vx
                settings.viy[n] = vy
                settings.viz[n] = vz
                n += 1
                
                nz += 1
            
            ny += 1
        
        nx += 1
    settings.nparticles = n
    print("in initialization: n = ",n)
        
    if settings.debug == 1:
        debug.WriteAtoms(0)
        
    # cancel the linear momentum
    svx = np.sum(settings.vix)
    svy = np.sum(settings.viy)
    svz = np.sum(settings.viz)
    
    settings.vix -= svx / settings.nparticles
    settings.viy -= svy / settings.nparticles
    settings.viz -= svz / settings.nparticles
    
    svx = np.sum(settings.vix)
    print("========== linear momentum ", svx)
    
    # rescale the velocity to the desired temperature
    Trandom = temperature()
    
        
    # rescale velocity to the desired temperature
    #settings.vix = rescalevelocity(settings.Tdesired, Trandom)
    rescalevelocity(settings.Tdesired, Trandom)
    
    Trandom = temperature()
    print("temperature == ", Trandom) 
   

def temperature():
    
    
    vsq = 0.
    vsq = np.sum(np.multiply(settings.vix, settings.vix) + np.multiply(settings.viy, settings.viy) + np.multiply(settings.viz, settings.viz))
    return settings.mass * vsq / 3./settings.kb / settings.nparticles
    
    
def rescalevelocity(T1, T2):
    
    settings.vix = settings.vix * math.sqrt(T1 / T2)
    settings.viy = settings.viy * math.sqrt(T1 / T2)
    settings.viz = settings.viz * math.sqrt(T1 / T2)
                          
    
    
    