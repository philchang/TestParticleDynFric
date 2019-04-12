import matplotlib.pyplot as pl
import numpy as np
import math

# imposes a symmetrized cut
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

def filterAngle( angle) :
    twopi = 2.*math.pi
    for i in range(angle.size - 1) :
        curAngle = angle[i]
        nextAngle = angle[i+1]
        dangle = abs(nextAngle-curAngle)
        if( abs( nextAngle + twopi - curAngle) < dangle ) : 
            angle[i+1:] += twopi
        elif( abs( nextAngle - twopi - curAngle) < dangle ) : 
            angle[i+1:] -= twopi
    return angle 


phi = np.loadtxt("phigrid.txt")
r = np.loadtxt( "rgrid.txt")
sigma = np.loadtxt( "sigma.txt",delimiter=',')
r /= 1000 # convert to kpc

fftArray = []
for i in range(r.size) : 
    xDisk = r[i]*np.cos( phi)
    yDisk = r[i]*np.sin( phi)

    # symmetrize the cut
    cutArray = cut(xDisk,yDisk,rin=0.)
    sigma[np.logical_not(cutArray),i] = 0.

    # compute fft per radius
    fftTheta = np.fft.rfft(sigma[:,i])
    fftArray.append(fftTheta)

fft = np.array(fftArray)

pl.clf()
ameff = []
rStart = 10
rEnd = 30

for m in range( 1, 5) :
    am = np.abs(fft[:,m])/np.abs(fft[:,0])
    pl.plot( r, am,label="$m={0}$".format(m))
    #effectively performs the integral
    ameff.append( np.mean( am[np.logical_and(r>rStart, r<rEnd)]))
ameff = np.array(ameff)
print(ameff) #print ameff for m=1,2,3,4

#compute the two defintions of ateff
ateff_data = math.sqrt(np.mean(ameff**2))
ateff_m13 = math.sqrt( np.mean(ameff[[0,2]]**2))
print( "Old Definition of ateff = {0:.3e}, for m=1,3 modes ateff = {1:.3e}".format(ateff_data, ateff_m13))

pl.legend(loc="best")
pl.savefig( "ateff_data.pdf")

angle = np.angle(fft[:,1])#+np.pi
angle[angle < 0] += 2.*np.pi
angle = filterAngle( angle)

boolArray = np.logical_and( r < 25., r > 20.)
p = np.polyfit( r[boolArray], angle[boolArray], 1)
print("slope = {0} phi_offset = {1}".format( p[0], p[1]))

pl.clf()
pl.plot( r, angle/math.pi*180)
#pl.ylim(0,360)
pl.savefig( "phase_data.pdf")
