try :
    import mkl
    mkl.set_num_threads(4)
except :
    print("Not using intel/mkl")

import numpy as np
import scipy as sp
import math

iterations = 50000
thin = 200
discard = 5000
unique_name = True

excluded_pulsars = None
nanograv = [0,1,2,3]
ppta = [4,5,6,7,8,9]
epta = [10,11,12,13]
best_pulsars = [0, 12, 2, 3, 4, 5, 6, 9, 10, 11]
use_other_distance = [5, 7, 9]
pulsars_number = None
pulsar_data = None

# model
QUILLEN = 0
EXPONENTIAL = 1
GAUSSIAN = 2
number_parameters = 1 # number of parameters for the galactic model

MODEL = QUILLEN
if MODEL == QUILLEN : 
    number_parameters = 2
elif MODEL == EXPONENTIAL or MODEL == GAUSSIAN :
    number_parameters = 2

## location of Sun -
rsun= 8.122 ## in kpc
xsun = rsun
ysun = 0.
zsun = 0.0055  ## in kpc from Quillen et al. 2020
kpctocm = 3.086e21 ## convert pc to cm
pc = kpctocm*1e-3

day_to_sec = 24*3600
c = 3e10

#Quillen model constants
alpha1_0 = 4e-30
alpha2_0 = -1e-51
lgalpha1_0 = math.log10( alpha1_0)
lgalpha2_0 = math.log10( -alpha2_0)

p1_range = 5
p2_range = 2

# exponential or gaussian model constants
rho_midplane = 0.1 # solar mass per cubic parsec
rho_midplane *= 2e33/(pc**3)
lgrho_midplane = math.log10(rho_midplane)
scale_height = 0.1*kpctocm # 500 pc
lgscale_height = math.log10(scale_height)


xsun *= kpctocm
ysun *= kpctocm
zsun *= kpctocm

mas_per_year_to_as_per_sec = 1e-3/3.15e7

Vlsr_quillen = 233.e5 ## from Schonrich et al. 2010 in cm/s

Vlsr0 = Vlsr_quillen#
Vlsr0 = 255.2*1e5
Vlsr_err = 5.1*1e5
G = 6.67e-8

def acc_gauss(x,y,z,rho0,z0, Vlsr) : 
    r = np.sqrt(x*x + y*y) 
    az = -4*np.pi*G*rho0*z0*sp.special.erf(np.abs(z)/z0)*np.sign(z)*math.sqrt(math.pi)*0.5
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def acc_exp(x,y,z,rho0,z0, Vlsr) : 
    r = np.sqrt(x*x + y*y) 
    az = -4*np.pi*G*rho0*z0*(1.-np.exp(-np.abs(z)/z0))*np.sign(z)
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def acc_quillen(x,y,z,lgalpha1, lgalpha2, Vlsr) :
    r = np.sqrt(x*x + y*y) 
    alpha1 = 1e1**lgalpha1
    alpha2 = 0.
    if number_parameters > 1 : 
        alpha2 = -1e1**lgalpha2
    az = -alpha1*z - alpha2*(z*z)*np.sign(z)
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r

    return ax,ay,az

## define acceleration components to fit the data, from Quillen et al. 2020:
def alos(x,y,z,parameter1,parameter2, Vlsr):

    ax,ay,az = 0,0,0
    axsun, aysun, azsun = 0,0,0
    if MODEL == QUILLEN : 
        lgalpha1, lgalpha2 = parameter1, parameter2
        axsun, aysun, azsun = acc_quillen(xsun,ysun,zsun, lgalpha1, lgalpha2, Vlsr)
        ax, ay, az = acc_quillen(x,y,z,lgalpha1,lgalpha2, Vlsr)
    elif MODEL == EXPONENTIAL :
        lgrho0, lgz0 = parameter1, parameter2
        rho0 = 1e1**lgrho0
        z0 = 1e1**lgz0
        axsun, aysun, azsun = acc_exp(xsun,ysun,zsun, rho0, z0, Vlsr)
        ax, ay, az = acc_exp(x,y,z,rho0,z0, Vlsr)
    elif MODEL == GAUSSIAN :
        lgrho0, lgz0 = parameter1, parameter2
        rho0 = 1e1**lgrho0
        z0 = 1e1**lgz0
        axsun, aysun, azsun = acc_gauss(xsun,ysun,zsun, rho0, z0, Vlsr)
        ax, ay, az = acc_gauss(x,y,z,rho0,z0, Vlsr)

    dx, dy, dz = x-xsun, y-ysun, z-zsun
    dr = np.sqrt(dx*dx+dy*dy+dz*dz)

    return ((ax-axsun)*dx + (ay-aysun)*dy + (az-azsun)*dz)/dr

def alos_from_pulsar(d, b, l, mus, lgalpha1, lgalpha2, Vlsr) : 
    ## pulsar positions --
    x = d * np.cos(b*np.pi/180.) * np.cos(l*np.pi/180.) *kpctocm + xsun
    y = d * np.cos(b*np.pi/180.) * np.sin(l*np.pi/180.) * kpctocm + ysun
    z = d * np.sin(b*np.pi/180.) * kpctocm + zsun

    al = alos(x,y,z,lgalpha1,lgalpha2, Vlsr)

    # include sloskey effect mu^2 *d /c
    al_sl = mus*mus*d*kpctocm/c

    return al+al_sl

def string_to_value_and_error( string, has_exponent=False) :
    import re
    from decimal import Decimal
    substring = r"(-*\d+\.\d+)\((\d+)\)"
    if( has_exponent) :
        substring = r"(-*\d+\.\d+)\((\d+)\)e(-\d+)$"

    value = re.sub(substring,r"\1", string)
    error = re.sub(substring,r"\2", string)
    err_exp = Decimal(value).as_tuple().exponent
    if( has_exponent) :
        exp = re.sub(substring, r"\3", string)
        exponent = 10.**float(exp)
        return float(value)*exponent, float(error)*10**err_exp*exponent
    return float(value), float(error)*10**err_exp

def initialize_theta( frac_random=1) :
    global pulsar_data 
    number_pulsars = pulsar_data["number pulsars"]
    distances = pulsar_data["distance"]
    distance_err = pulsar_data["distance error"]
    mus = pulsar_data["mu"]
    mu_err = pulsar_data["mu error"]
    theta = np.zeros(number_parameters+2*number_pulsars)
    if MODEL == QUILLEN : 
        lgalpha1 = lgalpha1_0 + 0.1*np.random.randn(1)[0]*frac_random
        lgalpha2 = lgalpha2_0 - 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgalpha1
        if(number_parameters > 1) :
            theta[1] = lgalpha2
    elif MODEL == EXPONENTIAL or MODEL == GAUSSIAN: 
        lgrho0 = lgrho_midplane + 0.1*np.random.randn(1)[0]*frac_random
        lgz0 = lgscale_height + 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgrho0
        theta[1] = lgz0

    if(number_parameters > 2) :
        theta[2] = Vlsr0 + Vlsr_err*np.random.randn(1)[0]*frac_random
    theta[number_parameters:number_pulsars+number_parameters] = distances + distance_err*np.random.randn(number_pulsars)*frac_random
    theta[number_pulsars+number_parameters:] = mus + mu_err*np.random.randn(number_pulsars)*frac_random
    return theta

def unpack_theta( theta, number_pulsars) :
    parameter1 = theta[0]
    parameter2 = -np.inf

    if( number_parameters > 1) :
        parameter2 = theta[1]

    Vlsr = Vlsr0

    if( number_parameters > 2) : 
        Vlsr = theta[2]
    distances = theta[number_parameters:number_pulsars+number_parameters]
    mus = theta[number_parameters+number_pulsars:]
    return parameter1, parameter2, Vlsr, distances, mus

def log_likelihood( theta) :
    global pulsar_data
    number_pulsars = pulsar_data["number pulsars"]
    lgalpha1, lgalpha2, Vlsr, distances, mus = unpack_theta(theta, number_pulsars)
    b = pulsar_data["latitude"]
    l = pulsar_data["longitude"]
    alos_model = alos_from_pulsar( distances, b, l, mus, lgalpha1, lgalpha2, Vlsr)
    alos_obs = pulsar_data["alos"]
    alos_err = pulsar_data["alos error"]

    ll = 0.5*np.sum( -((alos_obs - alos_model)/alos_err)**2 - np.log(2*np.pi*alos_err*alos_err))
    return ll

def log_prior( theta) :
    global pulsar_data 
    number_pulsars = pulsar_data["number pulsars"]
    p1, p2, Vlsr, distances, mus = unpack_theta(theta, number_pulsars)

    distance_err = pulsar_data["distance error"]
    mu_err = pulsar_data["mu error"]
    distance_arr = pulsar_data["distance"]
    mu_arr = pulsar_data["mu"]

    # define the range in alpha1, alpha2
    lp = 0
    if MODEL == QUILLEN : 
        lgalpha1, lgalpha2 = p1, p2
        if( np.abs(lgalpha1 - lgalpha1_0) > p1_range)  :
            return -np.inf
        if(number_parameters > 1 and np.abs(lgalpha2 - lgalpha2_0) > p2_range) :
            return -np.inf
    elif MODEL == EXPONENTIAL or MODEL == GAUSSIAN :
        lgrho0, lgz0 = p1,p2 
        if( np.abs(lgrho0 - lgrho_midplane) > p1_range)  :
            return -np.inf
        if(number_parameters > 1 and np.abs(lgz0 - lgscale_height) > p2_range) :
            return -np.inf

    #lp += -0.5*(((Vlsr - Vlsr0)/Vlsr_err)**2 + math.log(2*math.pi*Vlsr_err**2))
    lp += 0.5*np.sum(-((distances-distance_arr)/distance_err)**2 - np.log(2*np.pi*distance_err**2))
    lp += 0.5*np.sum(- ((mus-mu_arr)/mu_err)**2 - np.log(2*np.pi*mu_err**2))
    return lp

def log_probability(theta) :
    global pulsar_data
    lp = log_prior( theta)
    if not np.isfinite( lp) : 
        return -np.inf
    return lp + log_likelihood( theta)


import pandas as pd

# import the data and decode
dataframe = pd.read_excel("PBDOT.xls") 
distances = dataframe['Parallax Distance (kpc)'] 
names = dataframe["Pulsar Name"]
datasets = dataframe["Dataset"]
mus = dataframe["Proper Motion (mu, mas/yr)"]
pbs = dataframe["PB (d)"]
pbdot_obs = dataframe["PBDOT_obs (s/s)"]
longitude = dataframe["Galactic Longitude (l, deg)"]
latitude = dataframe["Galactic Latitude (b, deg)"]
other_distances = dataframe["Other Distance (kpc)"]

name_arr = []
dataset_arr = []
distance_arr = []
distance_err = []
latitude_arr = []
longitude_arr = []
mu_arr = []
mu_err = []

alos_arr = []
alos_err = []

for name, dataset, distance_str, other_distance_str, mu_str, pb_str, pbdot_str, lat_str, long_str, pulsar_no in \
        zip(names, datasets, distances, other_distances, mus, pbs, pbdot_obs, latitude, longitude, range(names.size)):
    if( (unique_name and name in name_arr) or (not excluded_pulsars is None and name in excluded_pulsars)) : 
        continue
    if( not pulsars_number is None and not pulsar_no in pulsars_number) :
        continue
    if( not best_pulsars is None and not pulsar_no in best_pulsars) :
        continue
    name_arr.append(name) 
    dataset_arr.append(dataset)
    d, derr = string_to_value_and_error(distance_str)

    if(pulsar_no in use_other_distance) : 
        d, derr = string_to_value_and_error(distance_str)
    mu, muerr = string_to_value_and_error( mu_str)
    mu *= mas_per_year_to_as_per_sec
    muerr *= mas_per_year_to_as_per_sec
    pbdot, pbdoterr = string_to_value_and_error( pbdot_str, has_exponent=True) 
    pb, pberr = string_to_value_and_error( pb_str)
    pb *= day_to_sec

    distance_arr.append(d)
    distance_err.append( derr)
    mu_arr.append( mu)
    mu_err.append( muerr)
    alos_arr.append(pbdot/pb*c)
    alos_err.append(pbdoterr/pb*c)
    latitude_arr.append(float(lat_str))
    longitude_arr.append(float(long_str))
    print(name, dataset)

longitude_arr = np.array(longitude_arr)
latitude_arr = np.array(latitude_arr)
distance_arr = np.array(distance_arr)
distance_err = np.array(distance_err)
mu_arr = np.array(mu_arr)
mu_err = np.array(mu_err)
alos_arr = np.array(alos_arr)
alos_err = np.array(alos_err)

pulsar_data = {"name" : name_arr, "dataset": dataset_arr, "longitude": longitude_arr, "latitude" : latitude_arr, \
                "distance" : distance_arr, "distance error" : distance_err, "mu" : mu_arr, "mu error" : mu_err, \
                "alos" : alos_arr, "alos error" : alos_err, "number pulsars" : len(name_arr)}


import emcee

itheta = initialize_theta( frac_random=0.)

nwalkers = 100
ndim = itheta.size

pos = np.zeros( [nwalkers,ndim])

for i in range(nwalkers) : 
    pos[i,:] = initialize_theta()

import multiprocessing

#with multiprocessing.Pool(1) as pool : 
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, iterations, progress=True)

samples = sampler.get_chain()
import matplotlib.pyplot as pl
# pl.clf()
# fig, axes = pl.subplots(3, figsize=(10, 7), sharex=True)
# labels = ["lgalpha1", "lgalpha2", "d"]
# for i in range(3):
#     ax = axes[i]
#     ax.plot(samples[:, :, i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)

# pl.savefig("test.pdf")


flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
print(flat_samples.shape)

import corner

pl.clf()
labels = [r"$\log(\alpha_1)$",r"$\log(-\alpha_2)$", r"$V_{\rm lsr}$"]
if MODEL == EXPONENTIAL or MODEL == GAUSSIAN : 
    labels = [r"$\log(\rho_0/1\,M_{\odot}\,{\rm pc}^{-3})$",r"$\log(z_0/1\,{\rm pc})$", r"$V_{\rm lsr}$"]
    flat_samples[:,0] -= math.log10(2e33/pc**3)
    flat_samples[:,1] -= math.log10(pc)
fig = corner.corner( flat_samples[:,0:number_parameters], labels=labels[0:number_parameters],truths=itheta[0:number_parameters])
pl.savefig("corner.pdf")

for i in range(number_parameters):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    print( "{0} = {1} {2} {3}".format(labels[i], mcmc[1], q[0], q[1]))

tau = sampler.get_autocorr_time()
print(tau)