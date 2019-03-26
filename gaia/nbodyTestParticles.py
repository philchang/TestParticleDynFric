import numpy as np
from scipy.integrate import ode
import argparse
import math

try :
    import matplotlib
    matplotlib.use("Agg")
except ImportError : 
    print( "Running without matplotlib")

import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gp
from galpy.util import bovy_conversion 
import scipy
import sys 


useMPI = True
try :
    from mpi4py import MPI
except ImportError:
    useMPI = False

rank = 0
size = 1
comm = None
if useMPI : 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2 : 
        useMPI = False
    else : 
        if rank == 0 :
	    print( "Running with MPI") 

#from numba import jit 

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--show_plot', action='store_true', help='make plots (r.pdf, xy.pdf, z.pdf)')
#parser.add_argument('--forward', action='store_true', help='forward in time integration')
parser.add_argument('--use_MW', action='store_true', help='MW Potential')
parser.add_argument('--use_MW14', action='store_true', help='MW14 Potential')
parser.add_argument('--use_K13', action='store_true', help='K13 NFW MW values')
parser.add_argument('--use_NFW', action='store_true', help='NFW Potential')
parser.add_argument('--use_Hernquist', action='store_true', help='NFW Potential')
parser.add_argument('--use_DF', action='store_true')
parser.add_argument('--use_TestParticles', action='store_true')
parser.add_argument('--run_test', action='store_true')
args = parser.parse_args()

useNFW = args.use_NFW or args.use_K13
useHernquist = args.use_Hernquist
useDF = args.use_DF
useTestParticles = args.use_TestParticles

MWpot = None
if args.use_MW : 
    MWpot = gp.MWPotential
elif args.use_MW14 : 
    MWpot = gp.MWPotential2014

pc = 3.0857e18
kpc = 1e3*pc
vkms = 1e5
msun = 1.989e33
G = 6.67e-8
m = 1.
TINY = 1e-30
yrs = 365.24*24*3600
Myrs = 1e6*yrs
Gyrs = 1e3*Myrs

# MW parameters
Mvir = 1.5e12*msun
cvir = 9.56
Rvir = 299.*kpc

# for hernquist
M0 = 1.25e12*msun
vc200=1.6e7
c0=9.39

# satellite parameters for hernquist
Nsat = 1
msat = np.zeros(Nsat)
c0sat = np.zeros(Nsat)
msat[0] = 1e10*msun
vc200sat = vc200*(msat/M0)**0.333e0
c0sat[:] = 9.39 

# test Particle parameters
nDisk = 100000
rbins = 200
nSatellite  = 10000

tEnd = 0.*Gyrs # in the future
tStart = -1.*Gyrs # in the past
tStep = 0.025*Gyrs

if args.use_K13 :
    # K13 values
    Rvir = 329.*kpc
    cvir = 9.36
    Mvir = 2.e12*msun

r0 = 8.0 # r0 8kpc
v0 = 220. # rotation rate

def derivs(t, y):
    N = y.size/6
    arr = y.reshape(2,N,3)
    r = arr[0,:,:]
    v = arr[1,:,:]
    a, dt = calAcc( r, v)
    return np.array([v,a]).flatten()

def findTimeStep( r,v) : 
    a, dt = calAcc( r, v)
    return 0.1*dt.min()

def getTestICs(nn=20,rin=5,rout=40) : 
    x = np.zeros(nn) 
    y = np.arange(rin,rout,(rout-rin)/nn)*kpc
    z = np.zeros(nn)

    vx = -500.*vkms*np.ones(nn)
    vy = np.zeros(nn)
    vz = np.zeros(nn)

    positions = []
    velocities = []

    for i in range(nn) : 
        positions.append( [[x[i],y[i],z[i]]])
        velocities.append( [[vx[i],vy[i],vz[i]]])
    positions = np.array( positions)
    velocities = np.array( velocities)

    return positions,velocities

def getSpecificICs() :
    x = -31.23593361*kpc 
    y = 128.61252308*kpc 
    z = -5.5784614*kpc

    vx = 57.78276892*vkms
    vy = -107.27936341*vkms 
    vz = -7.88558141*vkms

    positions = np.array( [[x,y,z]])
    velocities = np.array([[vx,vy,vz]])
    return -1*Gyrs, positions, velocities

def getICs(pm_ra_cosdec=-0.095,pm_dec=0.058,radial_velocity=290.7) : # return ICs in galactocentric coordinates
    distance = 129.4e3*u.pc 

    c1 = coord.ICRS(ra=143.8868*u.degree, dec=-36.7673*u.degree,
                    distance=129.4e3*u.pc,
                    pm_ra_cosdec=pm_ra_cosdec*u.mas/u.yr,
                    pm_dec=pm_dec*u.mas/u.yr,
                    radial_velocity=radial_velocity*u.km/u.s)

    gc1 = c1.transform_to(coord.Galactocentric)

    vx = gc1.v_x
    vy = gc1.v_y
    vz = gc1.v_z

    l = 264.8955
    b = 11.2479 
    R0 = r0*1.e3*u.pc

    #convert to Galactocentric -- 

    x = distance * np.cos(b*np.pi/180.) * np.cos(l*np.pi/180.) - R0
    y = distance * np.cos(b*np.pi/180.) * np.sin(l*np.pi/180.) 
    z = distance* np.sin(b*np.pi/180.)

    positions = np.array( [[x.value,y.value,z.value]])*pc
    velocities = np.array([[vx.value,vy.value,vz.value]])*vkms

    return positions,velocities

def getVelSamples( nn=10) : 

    vavg = [-80.93960285062732,-48.51070213345165, 50.56259604843662]
    sigmax = 22.4883097
    sigmay = 231.308893
    sigmaz = 64.9421574

    svx = np.random.normal(vavg[0], sigmax, nn)
    svy = np.random.normal(vavg[1], sigmay, nn)
    svz = np.random.normal(vavg[2], sigmaz, nn)

    velocities = np.array(list(zip(svx,svy,svz)))*vkms
    return velocities

def getICsmusample(nn=10):
    distance = 129.4e3*u.pc

# define pm samples:
    spmracosdec = np.random.normal(-0.095, 0.018, nn)
    spmdec = np.random.normal(0.058, 0.024, nn)

    vx = np.empty(nn)
    vy = np.empty(nn)
    vz = np.empty(nn)
    positions = []
    velocities = []
    for pm_ra_cosdec, pm_dec in zip( spmracosdec, spmdec):
        pos, vel = getICs(pm_ra_cosdec=pm_ra_cosdec,pm_dec=pm_dec)
        positions.append( pos)
        velocities.append( vel)

    return positions,velocities

def NFWmass( r) : 
    scalelength = Rvir/cvir
    x = r/scalelength
    xRv = Rvir/scalelength
    mass0 = Mvir/(math.log(1. + cvir) - cvir/(1.+cvir))

    fcvir = math.log(1.+cvir)-(cvir/(1.+cvir))
    rhos = Mvir/(4*math.pi*(scalelength**3)*fcvir)

    #following Zentner & Bullock 03, section 2.2, eqn 3                                                                      

    Vvir = math.sqrt(G*Mvir/Rvir)

    Vmaxsq = 0.216*(Vvir**2)*(cvir/fcvir)
    Vmax = math.sqrt(Vmaxsq)

    rs = Rvir/cvir
    xsigma = r/rs  # Zentner & Bullock 2003, equation 6
   
    sigma = Vmax*((1.439*(xsigma**0.354))/(1.+1.1756*(xsigma**0.725)))

    if (r < Rvir) : 
        m = mass0*(math.log(1.+x) - (x/(1.+x)))
        rho = rhos/(x*((1.+x)**2))

    else : 
        m=(mass0*(math.log(1+xRv) - (xRv/(1.+xRv))))
        rho = 0.

    return m,rho,sigma

def hernquistmass( r, m0 = M0, vc200=vc200, c0=c0) : 
    r200 = G*m0/(vc200*vc200)
    rs = r200/c0
    a = rs*math.sqrt(2.*(math.log(1.+c0) - c0/(1.+c0)))
    m = m0 * r*r/((r+a)*(r+a))
    rho = (m0/(2.*math.pi))*(a/(r*(r+a)**3.))
    r=r200
    Vmax = 215.*vkms  # for V200 = 160                                                                                 
    rs = r200/c0
    x = r/rs  # Zentner & Bullock 2003, equation 6                                                                      
    sigma = Vmax*((1.439*(x**0.354))/(1.+1.1756*(x**0.725)))
    return m, rho, sigma

def calAcc(r, v) : 
    N = r.shape[0]
    dr = r
    dx = r[:,0]
    dy = r[:,1]
    dz = r[:,2]
    dv = v
    dr_gp = np.sqrt(dx*dx + dy*dy)/(r0*kpc) # galpy units
    dz_gp = dz/(r0*kpc)
    drNorm = np.linalg.norm(dr, axis = 1)
    acc = None
    if useNFW : 
        m, rho, sigma = NFWmass( drNorm)
        acc = -G*m/(drNorm*drNorm)[:,np.newaxis]*dr/drNorm[:,np.newaxis]
    elif useHernquist:
        m, rho, sigma = hernquistmass( drNorm)
        acc = (-G*m/(drNorm*drNorm*drNorm))[:,np.newaxis]*dr#[:,np.newaxis]*dr
    else : 
        acc = np.zeros([dr_gp.size,3])
        conv = 1e-13*1e5
        for i, x_gp, y_gp, r_gp, z_gp in zip(range(dr_gp.size), dx/(r0*kpc), dy/(r0*kpc), dr_gp, dz_gp) : 
            a = gp.evaluateRforces(MWpot,r_gp,z_gp)*bovy_conversion.force_in_10m13kms2(v0,r0)*conv
            acc[i,0] = a*x_gp/r_gp
            acc[i,1] = a*y_gp/r_gp
            acc[i,2] = gp.evaluatezforces(MWpot,r_gp,z_gp)*bovy_conversion.force_in_10m13kms2(v0,r0)*conv
    dt_rmin = np.min(drNorm/np.linalg.norm(dv, axis=1))
    dt_amin = np.min(np.linalg.norm(dv,axis=1)/np.maximum(np.linalg.norm(acc,axis=1), TINY))
    dt_min = min(dt_rmin, dt_amin)
    if(useDF) : 
        acc[:Nsat,:] = acc[:Nsat,:] + accDF( r[:Nsat,:], v[:Nsat,:])
    if(useTestParticles and Nsat < N) :
        # include accelerations from other satellites for test particles
        for i in range(Nsat) : 
            rsat = r[i]
            dr = r[Nsat:,:] - rsat[np.newaxis,:]
            drNorm = np.linalg.norm( dr, axis=1)
            m, rho, sigma = hernquistmass( drNorm, m0=msat[i], vc200=vc200sat[i], c0=c0sat[i])
            acc[Nsat:,:] =  acc[Nsat:,:] - (G*m/(drNorm*drNorm*drNorm))[:,np.newaxis]*dr
    return acc, dt_min

def accDF( r, vel, msat=msat) : 
    if not (useNFW or useHernquist) :
        return np.zeros(r.shape)
    
    dr = r
    drNorm = np.linalg.norm(dr, axis = 1)
    
    m = 0. 
    rho = 0.
    sigma = 0.
    if useNFW : 
        m,rho,sigma = NFWmass( drNorm)
    elif useHernquist:
        m,rho,sigma = hernquistmass( drNorm)

    v = np.linalg.norm( vel, axis=1)

    xchandra = v/(math.sqrt(2.)*sigma)
    erf = scipy.special.erf(xchandra)
    # following Besla et al. 2007 here.  TODO: vary bmin as function of perturber mass.
    Lambda = drNorm/(3.*1.6*kpc)   
    #to avoid problem when r < Rsat
    logLambda=np.maximum(0.,math.log(Lambda))  

    accdf = (-4.*math.pi*G**2*(logLambda)*rho*(1./v**3)*(erf-2*xchandra*(1./math.sqrt(math.pi))*(math.exp(-(xchandra**2))))*msat)[:,np.newaxis]*vel
    return accdf

def getStartingPosition(pos, vel, tEnd = tStart) :
    #print( "Initial positions: {0} and velocities: {1}".format(pos/kpc, vel/vkms))
    N = pos.shape[0]

    t = 0.
    y0 = np.array([pos,vel]).flatten()
    integral = ode( derivs).set_integrator("dop853")
    integral.set_initial_value(y0, t)

    tArray = []
    rArray = []
    xArr = []
    yArr = []
    zArr = []

    while integral.successful() and tEnd < t :
        dt = max(-findTimeStep(pos,vel),tEnd-t)
        y = integral.integrate( integral.t+dt)
        t = integral.t
        arr = y.reshape(2,N,3)
        pos = arr[0,:,:]
        vel = arr[1,:,:]
        r = np.linalg.norm( pos, axis=1)
        xArr.append(pos[0,0]/kpc)
        yArr.append(pos[0,1]/kpc)
        zArr.append(pos[0,2]/kpc)
        tArray.append(t/Gyrs)
        rArray.append(r[0]/kpc)

    minr = np.array(rArray).argmin()
    #print( "Ending positions: {0} and velocities: {1}".format(pos/kpc, vel/vkms))

    return t, pos,vel

def setupDiskParticles( nDisk=nDisk, rbins=rbins, rin = 5.*kpc, rout=30.*kpc) :
    thetaBins = int(nDisk/rbins)
    dr = (rout-rin)/rbins
    r = np.arange(rin+0.5*dr, rout+0.5*dr, dr)
    m, rho, sigma = hernquistmass(r)
    thetaStep = 2.*math.pi/thetaBins
    theta = np.arange(-math.pi+0.5*thetaStep, math.pi + 0.5*thetaStep, thetaStep)
    phase = 2.*math.pi*np.random.rand(rbins)
    numParticles = theta.size*r.size
    pos = np.zeros( [numParticles, 3])
    vel = np.zeros( [numParticles, 3])
    for i in range(rbins) : 
        cosx = np.cos( theta + phase[i])
        sinx = np.sin( theta + phase[i])
        pos[i*thetaBins:(i+1)*thetaBins,0] = r[i]*cosx
        pos[i*thetaBins:(i+1)*thetaBins,1] = r[i]*sinx
        v = math.sqrt(G*m[i]/(r[i]))
        vel[i*thetaBins:(i+1)*thetaBins,0] = -v*sinx
        vel[i*thetaBins:(i+1)*thetaBins,1] = v*cosx
    
    return pos, vel

def setupSatParticles( pos, vel, nSatellite=nSatellite, msat=msat[0], vcsat = vc200sat[0], c0sat = c0sat[0], rout=2.*kpc) :
    nRandom = int(nSatellite*1.2*8./(4.*math.pi/3.))
    x = 2.*rout*(np.random.rand(nRandom)-0.5)
    y = 2.*rout*(np.random.rand(nRandom)-0.5)
    z = 2.*rout*(np.random.rand(nRandom)-0.5)
    r = np.sqrt(x*x + y*y + z*z)
    posParticles = np.array(zip(x,y,z))[r<rout,:]
    posParticles = posParticles[0:nSatellite,:] + pos[np.newaxis,:]
    r = r[r<rout][0:nSatellite]
    m, rho, sigma = hernquistmass(r, m0=msat, vc200=vcsat, c0=c0sat)
    v = np.sqrt(G*m/r)
    vx = 2.*(np.random.rand(nSatellite)-0.5)
    vy = 2.*(np.random.rand(nSatellite)-0.5)
    vz = 2.*(np.random.rand(nSatellite)-0.5)
    vr = np.sqrt(vx*vx + vy*vy + vz*vz)

    vx = v*vx/vr
    vy = v*vy/vr
    vz = v*vz/vr

    velParticles = np.array(zip(vx,vy,vz)) + vel[np.newaxis,:]

    return posParticles, velParticles    

def fourierModes( xDisk, yDisk, rin=10, rout=30, rbins=20, phiBins=180) :
    r = np.sqrt(xDisk*xDisk + yDisk*yDisk) 
    theta = np.arctan2(yDisk/r, xDisk/r) # note that x,y are reverse for arctan2
    rArray = []
    fftArray = []
    for i in range(rbins) : 
        r1 = rin + (rout-rin)/rbins*i
        r2 = rin + (rout-rin)/rbins*(i+1)
        boolArray = np.logical_and( r>= r1, r < r2)
        thetaBins, edges = np.histogram( theta[boolArray],bins=phiBins,range=[-math.pi, math.pi])
        fftTheta = np.fft.rfft(thetaBins)
        rArray.append( 0.5*(r1+r2))
        fftArray.append( fftTheta[0:5])

    return np.array(rArray), np.array(fftArray)


rp = kpc*np.arange( 10., 50., 1.)
m, rho, sigma = hernquistmass( rp)
rhoMean = m/(4.*math.pi/3.*rp**3)
rscale = 0.5*kpc
vKick = np.sqrt(G*rhoMean)*rscale 
posSamples, velSamples = getICsmusample( nn = 50)
if( args.run_test) :
    posSamples, velSamples = getTestICs()
#posSamples, velSamples = getICsmusample( nn = 1)
#vdist = getVelSamples()
rperiArray = []
ateffArray = []
for posSat, velSat in zip( posSamples, velSamples) : 
    from timeit import default_timer as timer
    start = timer()
    # we have run it backwards -- now run forward
    t = 0.
    posSatInit = None
    velSatInit = None
    rStart = 12.
    rperi = 100.
    if( rank == 0) : 
        t, posSatInit, velSatInit = getStartingPosition( posSat, velSat)
        #t, posSatInit, velSatInit = getSpecificICs()
    
    if( useTestParticles) : 
        if( useMPI) : 
            data = None
            if rank == 0:
                data = np.array([posSatInit,velSatInit])

            data = comm.bcast(data, root=0) # broadcast the array from rank 0 to all others
            posSatInit = data[0]
            velSatInit = data[1]
            if rank == 0:
                data = np.array([t])

            data = comm.bcast(data, root=0) # broadcast the array from rank 0 to all others
            t = data[0]

        dt = 0

        posDisk, velDisk = setupDiskParticles()
        posTestSat, velTestSat   = setupSatParticles(posSatInit[0], velSatInit[0])

        Ntest = 0
        NtestDisk = 0
        NtestSat = 0
        useTestSat = False
        if useMPI : 
            if rank == 0:
                NtestDisk = posDisk.shape[0]
                posDisk = posDisk.reshape(size,NtestDisk/size,3)
                velDisk = velDisk.reshape(size,NtestDisk/size,3)
                if( useTestSat) :
                    NtestSat = posTestSat.shape[0]
                    posTestSat = posTestSat.reshape(size,NtestSat/size,3)
                    velTestSat = velTestSat.reshape(size,NtestSat/size,3)
                    Ntest = NtestDisk + NtestSat
                else : 
                    Ntest = NtestDisk

            posDisk = comm.scatter(posDisk, root=0)
            velDisk = comm.scatter(velDisk, root=0)
            if( useTestSat) :
                posTestSat = comm.scatter(posTestSat, root=0)
                velTestSat = comm.scatter(velTestSat, root=0)
        pos = np.append( posSatInit, posDisk, axis=0)
        vel = np.append( velSatInit, velDisk, axis=0)

        if( useTestSat) : 
            pos = np.append( pos, posTestSat, axis=0)
            vel = np.append( vel, velTestSat, axis=0)
        tStart = t
        t = 0. # needs to be zeroed again
        N = pos.shape[0] # reset to include test particles
        y0 = np.array([pos,vel]).flatten()
    
        integral = ode( derivs).set_integrator("dop853")
        integral.set_initial_value(y0, t)
        iFrame = 0
        iterations = 0
        isLast = False
        while t < tEnd-tStart :
            iterations += 1
            tNext = t + tStep
            while integral.successful() and t < tNext : 
                dt = min(findTimeStep(pos,vel),tNext-t)
                y = integral.integrate( integral.t+dt)
                t = integral.t
                arr = y.reshape(2,N,3)
                pos = arr[0,:,:]
                vel = arr[1,:,:]
                posSat = pos[:Nsat]
                velSat = vel[:Nsat]
                r = np.linalg.norm( posSat, axis=1)
                rperi = min(rperi, r[0]/kpc)
                #print(t/Gyrs, rperi, r[0]/kpc)
                isLast = t == tEnd-tStart
            
            if useMPI : 
                data1 = comm.gather(pos[Nsat:,:], root=0)
                data2 = comm.gather(vel[Nsat:,:], root=0)
                if( rank == 0 ) :
                    pos = np.append(posSat,np.array(data1).reshape(Ntest,3),axis=0)
                    vel = np.append(velSat,np.array(data2).reshape(Ntest,3),axis=0)

            if( args.show_plot and rank == 0 and isLast) : 
            #if( args.show_plot and rank == 0) : 
                import matplotlib.pyplot as pl
                from scipy.stats.kde import gaussian_kde
                posTest = pos[Nsat:]
                velTest = vel[Nsat:]
                xDisk = posTest[:,0]/kpc
                yDisk = posTest[:,1]/kpc
                xSat = posSat[:,0]/kpc
                ySat = posSat[:,1]/kpc
                #xDisk = 70*(np.random.rand(1000000) - 0.5)
                #yDisk = 70*(np.random.rand(1000000) - 0.5)
                r, fft = fourierModes(xDisk, yDisk)
                iFrame = iFrame + 1

                #endSuffix = "{0:04d}.png".format(iFrame)
                endSuffix = "{0:05.2f}.png".format(rperi)

                pl.clf()
                fig = pl.figure()
                ax = fig.add_subplot(111)
		for i in range(1,fft.shape[1]) :
                    ax.plot( r, np.abs(fft[:,i])/np.abs(fft[:,0]), lw=2,label="m={0}".format(i))
                am = np.abs(fft[:,1:])/np.abs(fft[:,0])[:,np.newaxis]
                deltar = 10.
                dr = r[1]-r[0]
                ameff = am[np.logical_and(r>=rStart,r<rStart+deltar),:].sum(axis=0)*dr/deltar
                ateff = 2*math.sqrt((ameff*ameff).mean())
                rperiArray.append( rperi)
                ateffArray.append( ateff)
                ax.set_xlim(10,25)
                ax.set_ylim(0,0.4)
                ax.text(12,0.2, "t={0:.3f} Gyrs".format((t+tStart)/Gyrs), fontsize=14)
                plotName = "fft_frame{0}".format(endSuffix)
                print( "making plot: {0}".format(plotName))
                pl.legend(loc="best")
                pl.savefig( plotName)
                pl.close(fig)

                pl.clf()
                fig = pl.figure()
                ax = fig.add_subplot(111)
                ax.plot( r, np.angle(fft[:,1])+np.pi, lw=2)
                ax.set_xlim(10,25)
                plotName = "phase_frame{0}".format(endSuffix)
                print( "making plot: {0}".format(plotName))
                pl.savefig( plotName)
                pl.close(fig)

                pl.clf()
                fig = pl.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.scatter( xDisk[::10], yDisk[::10], s=1)
                ax.scatter( xSat, ySat, s=10, color="red")
                pl.xlabel( "x [kpc]", fontsize=20)
                pl.ylabel( "y [kpc]", fontsize=20)
                ax.set_xlim(-100,100)
                ax.set_ylim(-100,100)
                ax.text(8,31, "t={0:.3f} Gyrs".format((t+tStart)/Gyrs), fontsize=14)
                plotName = "frame{0}".format(endSuffix)
                print( "making plot: {0}".format(plotName))
                pl.savefig( plotName)
                pl.close(fig)
                '''
                pl.clf()
                fig = pl.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                #ax.scatter( xDisk-xSat, yDisk-ySat, s=1)
                ax.set_xlim(-4,4)
                ax.set_ylim(-4,4)
                pl.xlabel( "x [kpc]", fontsize=20)
                pl.ylabel( "y [kpc]", fontsize=20)
                ax.text(1.0,3.5, "t={0:.3f} Gyrs".format((t+tStart)/Gyrs), fontsize=14)

                plotName = "movie_frame{0}".format(endSuffix)
                print( "making plot: {0}".format(plotName))
                pl.savefig( plotName)
                pl.close(fig)
                '''
    end = timer()
    if rank == 0 :
       print("time of calculation {0:.2f}s".format(end - start)) # Time in seconds, e.g. 5.38091952400282
    if rank == 0 :
       import matplotlib.pyplot as pl
       pl.clf()
       pl.scatter( rperiArray, ateffArray, s=4)
       pl.xlim(0.,40.)
       pl.ylim(0.,0.35)
       pl.xlabel( r"$r_{p}$", fontsize=20)
       pl.ylabel( r"$a_{t,eff}$", fontsize=20)
       pl.xticks(fontsize=18)
       pl.yticks(fontsize=18)
       pl.tight_layout()
       pl.savefig("ateff.pdf")
