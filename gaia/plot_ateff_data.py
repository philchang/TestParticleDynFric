import matplotlib.pyplot as pl
import numpy as np
import math
import argparse

xlen = 40
SimulationFile = "Rp8nogas_xlen40_stars_020.dat".format(xlen)
runSimulationFile = True

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--cutPhil', action='store_true',
                    default=False,
                    help='sum the integers (default: find the max)')

parser.add_argument('--cutSukanya', action='store_true',
                    default=False,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

H0 = 1.0
# imposes a symmetrized cut
def cut( xDisk, yDisk, rin=-1, cutAngle = 15) :
    r = np.sqrt(xDisk*xDisk + yDisk*yDisk) 
    x0 = [8.5, -8.5]
    cutArray = None
    thetaCut1 = np.arctan2(yDisk, xDisk-x0[0])
    thetaCut2 = np.arctan2(yDisk, xDisk-x0[1])
    thetaCut1[thetaCut1 < 0] += 2.*math.pi
    #thetaCut2[thetaCut2 < 0] += 2.*math.pi
    cut = cutAngle/360.*2.*math.pi
    cutArray = np.logical_and( r > rin, np.logical_and( np.abs(thetaCut1 - np.pi) > cut,np.abs(thetaCut2 ) > cut))
    return cutArray

def cutSukanya( r, phi, sigma ) : 
    for i in range(r.size) : 
        minphi = phi[sigma[:,i] > 0].min()
        maxphi = phi[sigma[:,i] > 0].max()
        # reflect is across the other size
        cut = np.logical_and( phi > math.pi - minphi, phi < 3.*math.pi - maxphi)
        sigma[cut,i] = 0.        
    return sigma

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

def readSim( SimulationFile, xlen=xlen) :
    xpixels = 480
    ypixels = 480
    xmax = 1.*xlen
    xmin = -1.*xlen
    ymax = xmax
    ymin = xmin
    sigma = np.loadtxt( SimulationFile)
    sigma = sigma.flatten()
    sigma = sigma.reshape([xpixels,ypixels])

    # get r, phi positions
    x = np.zeros([xpixels,ypixels])
    y = np.zeros([xpixels,ypixels])
    r = np.zeros([xpixels,ypixels])
    phi = np.zeros([xpixels,ypixels])

    for j in range(ypixels) :
        for i in range(xpixels) :
            xpos = (xmax-xmin)/xpixels*(i +0.5)+xmin
            ypos = (ymax-ymin)/ypixels*(j +0.5)+ymin
            rpos = np.sqrt(xpos**2 + ypos**2)
            phipos = np.arctan2(ypos,xpos)
            x[j,i] = xpos
            y[j,i] = ypos
            r[j,i] = rpos
            if(phipos < 0.) :
                phipos += 2.*math.pi
            phi[j,i] = phipos

    print(sigma[sigma<0].shape)
    pl.clf()
    boolArr = sigma > 0
#    pl.scatter(x[boolArr].flatten(), y[boolArr].flatten())
    pl.hist2d(x.flatten(), y.flatten(), bins=480,weights=sigma.flatten())
    pl.savefig("testSim.png")

    nrgrid = 157
    nphigrid = 350
    rmax = 40
    rmin = 8
    rgrid = np.arange(rmin,rmax,(rmax-rmin)/nrgrid)
    phigrid = np.arange(0,2*math.pi,(2*math.pi)/nphigrid)
    ngrid = np.zeros([nphigrid, nrgrid])
    sigmarphi = np.zeros([nphigrid, nrgrid])
    for i in range( nrgrid) :
         
        rbool = r>= rgrid[i] 
        if( i+1 < nrgrid) :
            rbool = np.logical_and(rbool, r < rgrid[i+1])
        sigmaSubset = sigma[rbool]
        rSubset = r[rbool]
        phiSubset = phi[rbool]        
        for j in range( nphigrid) :
            phibool = phiSubset >= phigrid[j]
            if( j+1 < nphigrid) :
                phibool = np.logical_and(phibool, phiSubset < phigrid[j+1])
            sigmaSlice = sigmaSubset[phibool]
            if( sigmaSlice.size > 0) :
                sigmarphi[j,i] = sigmaSlice.mean()

    return rgrid, phigrid, sigmarphi
    

def readData() :
    phi = np.loadtxt("phigrid.txt")
    r = np.loadtxt( "rgrid.txt")
    sigma = np.loadtxt( "sigma.txt",delimiter=',')
    r /= 1000 # convert to kpc
    r *= H0
    return r,phi, sigma

r = None
phi = None
sigma = None
if( runSimulationFile) :
    r,phi,sigma = readSim( SimulationFile)
else :
    r,phi, sigma = readData()
pl.clf()
fig, ax = pl.subplots()
ax.set_aspect(1.0)
for i in range(sigma.shape[0]) :
    for j in range( sigma.shape[1]) : 
        if(math.isnan(sigma[i][j])) : 
            sigma[i][j] = 0.

sigma[sigma < 0.] = 0.
fftArray =[]

if( args.cutSukanya) : 
    sigma = cutSukanya(r, phi, sigma)

xDiskArr = []
yDiskArr = []
sigmaArr = []
for i in range(r.size) : 
    xDisk = r[i]*np.cos( phi)
    yDisk = r[i]*np.sin( phi)
    xDiskArr.append( xDisk)
    yDiskArr.append( yDisk)

    # symmetrize the cut
    cutAngle = -1 # don't include a cut
    if( args.cutPhil) :
        cutAngle = 15 # 15 degrees as specify in paper
        cutArray = cut(xDisk,yDisk,rin=0., cutAngle=cutAngle)
        sigma[np.logical_not(cutArray),i] = 0.
    sigmaArr.append(sigma[:,i])
    pl.scatter(xDisk[sigma[:,i]>0], yDisk[sigma[:,i]>0])

    # compute fft per radius
    fftTheta = np.fft.rfft(sigma[:,i])
    fftArray.append(fftTheta)


pl.savefig("HI.png") #plot the cuts.
fft = np.array(fftArray)
pl.clf()
xDiskArr = np.array(xDiskArr)
yDiskArr = np.array(yDiskArr)
sigmaArr = np.array(sigmaArr)

pl.hist2d(xDiskArr.flatten(), yDiskArr.flatten(), bins=200, weights= sigmaArr.flatten())
pl.savefig("sigma.png")
pl.clf()
ameff = []
rStart = 10
rEnd = 25

for m in range( 0, 5) :
    am = np.abs(fft[:,m])/np.abs(fft[:,0])
    pl.plot( r, am,label="$m={0}$".format(m))
    #effectively performs the integral
    ameff.append( np.mean( am[np.logical_and(r>rStart, r<rEnd)]))
ameff = np.array(ameff)

ameff *= 1.5 # multiply by 1.5 to keep consistency with Sukanya's scripts

for m in range(1,5) : 
    print( "a{0} = {1:.5e}".format( m, ameff[m]) )
#print(ameff*1.5) #print ameff for m=1,2,3,4

#compute the two defintions of ateff
ateff_data = math.sqrt(np.mean(ameff[1:]**2))
ateff_m13 = math.sqrt( np.mean(ameff[[1,3]]**2))
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
