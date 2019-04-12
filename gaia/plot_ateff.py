import matplotlib.pyplot as pl
import numpy as np
import math

def cut( xDisk, yDisk, rin=8, cutAngle = 15) :
    r = np.sqrt(xDisk*xDisk + yDisk*yDisk) 
    x0 = [8, -8]
    cutArray = None
    thetaCut1 = np.arctan2(yDisk, xDisk-x0[0])
    thetaCut2 = np.arctan2(yDisk, xDisk-x0[1])
    thetaCut1[thetaCut1 < 0] += 2.*math.pi
    #thetaCut2[thetaCut2 < 0] += 2.*math.pi
    cut = cutAngle/360.*2.*math.pi
    cutArray = np.logical_and( r > rin, np.logical_and( np.abs(thetaCut1 - np.pi) > cut,np.abs(thetaCut2 ) > cut))
    return cutArray

rp, ateff, slope, offset = np.loadtxt("ateff.data", unpack=True)

pl.scatter(rp, ateff, s=2, c="black")
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.xlabel(r"$r_{\rm p}$ [kpc]", fontsize=20)
pl.ylabel(r"$a_{\rm t,eff}$", fontsize=20)
pl.tight_layout()
pl.savefig("ateff.pdf")

pl.clf()
pl.scatter( rp, slope, s=2, c='black')
pl.plot( np.arange(20,25,0.1), 0*np.arange(20,25,0.1)-0.05, lw=2, color='red')
pl.savefig("phase.pdf")

