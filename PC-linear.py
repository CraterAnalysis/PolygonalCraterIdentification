"""
Created on Tue Jul 30 11:45:00 2019

@author: Stuart J. Robbins

This code takes in a crater rim trace, in decimal degrees, and determines if any
component of the trace meets set criteria for being considered an approximately
straight edge, and if any component between two edges meets criteria to be
considered a hinge.  It creates a graph showing the results and outputs the data
to the command line.


Modification Log:
  30-Jul-2019: -Initial port from Igor.
  31-Jul-2019: -Added movement of edges forward to search for better match, then
                rock back a bit to check for an extended start/end point.
               -Modified user output, including working timer.
               -Added smoothing factor based on testing Fekojoo crater (Ceres).
               -Fixed several bugs that caused crashes or infinite loops when
                dealing with edges going up to the end of the rim trace and
                angles between the first and last edge.
"""


##TO DO:  SEE "TO DO" ITEMS THROUGHOUT THIS FILE.



#Import the libraries needed for this program.
import argparse                     #allows parsing of input options
import numpy as np                  #lots of maths
import scipy as sp                  #lots of special maths
from scipy.signal import savgol_filter #specific maths!
import math                         #other lots of maths
import time                         #timing for debugging
import sys                          #for setting NumPY output for debugging
import matplotlib.pyplot as mpl     #for display purposes
import matplotlib.pyplot as plt     #for display purposes
from numpy import pi
import scipy.optimize
import scipy
import joblib

mem = joblib.Memory('.', verbose=False)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#Set any code-wide parameters.
np.set_printoptions(threshold=sys.maxsize) #for debugging purposes


#General help information.
parser = argparse.ArgumentParser(description='Identify polygonal crater rims.')

#Various runtime options.
parser.add_argument('--input',                              dest='inputFile',                       action='store', default='',     help='Name of the CSV file with a single crater rim trace with latitude,longitude in decimal degrees, one per line.')
parser.add_argument('--body_radius',                        dest='d_planet_radius',                 action='store', default='1',    help='Radius of the planetary body on which the crater is placed, in kilometers.')
#parser.add_argument('--tolerance_distance_min_forside',     dest='tolerance_distance_min_forside',  action='store', default='5',    help='The minimum length of a rim segment to be approximately straight to be considered an edge (in km).')
#parser.add_argument('--tolerance_distance_max_forhinge',    dest='tolerance_distance_max_forhinge', action='store', default='5',    help='The maximum length of a rim segment to be curved enough to be considered a hinge/joint between edges (in km).')
#parser.add_argument('--tolerance_angle_max_forside',        dest='tolerance_angle_max_forside',     action='store', default='10',   help='The maximum angle that a rim segment can vary for it to be considered a straight side (in degrees).')
#parser.add_argument('--tolerance_angle_min_forhinge',       dest='tolerance_angle_min_forhinge',    action='store', default='20',   help='The minimum standard deviation of the bearing angle a rim segment must vary within the given length for it to be considered a hinge/joint between edges (in degrees).')
#parser.add_argument('--smoothing_length',                   dest='smoothing_length',                action='store', default='1',    help='The rim trace will be smoothed using a Savitzky-Golay filter of a length in rim points where that length is divided by THIS value (e.g., 1 means the smoothing length is the average number of points corresponding to the minimum length of an edge).')

#Parse all the arguments so they can be stored
args=parser.parse_args()




##----------------------------------------------------------------------------##

#Store the time to output it at the end.
timer_start = time.time()



##-------------- SET UP THE DATA INTO A LOCAL COORDINATE SYSTEM --------------##


#Read the crater rim data.
rim_data = np.genfromtxt(args.inputFile,delimiter=',')

#Create some variables and vectors for use in transforms.  The center-of-mass
# variables are not standard because sometimes we do not have complete rims, and
# this helps get closer to the real center.
x_center_mass_degrees = (np.mean(rim_data[:,0])+(np.min(rim_data[:,0])+np.max(rim_data[:,0]))/2.)/2.
y_center_mass_degrees = (np.mean(rim_data[:,1])+(np.min(rim_data[:,1])+np.max(rim_data[:,1]))/2.)/2.
rim_lon_temp = np.copy(rim_data[:,0])
rim_lat_temp = np.copy(rim_data[:,1])
dist        = [0]*(len(rim_data))
bearing     = [0]*(len(rim_data))
atan2_part1 = [0]*(len(rim_data))
atan2_part2 = [0]*(len(rim_data))

#Determine the distance to each rim point from the {x,y} center of mass using
# Great Circles (Vincenty, 1975).
dist[:] = np.power(np.sin((rim_data[:,1]-y_center_mass_degrees)*math.pi/180/2),2) + np.cos(rim_data[:,1]*math.pi/180)*np.cos(y_center_mass_degrees*math.pi/180)*np.power((np.sin((rim_data[:,0]-x_center_mass_degrees)*math.pi/180/2)),2)
dist[:] = 2*np.arctan2(np.sqrt(dist[:]),np.sqrt(1.0 + np.negative(dist[:]))) * float(args.d_planet_radius)

#Calculate the positions of each rim point in {x,y} via calculating the angles
# between the original points and the center of mass, using bearings (Vincenty,
# 1975).  This is unfortunately a long set of equations so I have put it off
# into a separate function that returns the initial azimuth (i.e., the initial
# bearing between the center of mass and the end lat/lon; technically, bearing
# changes as you move along the Great Circle arc).
atan2_part1[:]  = np.sin((rim_lon_temp[:]-x_center_mass_degrees)*math.pi/180)*np.cos(rim_lat_temp*math.pi/180)
atan2_part2[:]  = np.cos(y_center_mass_degrees*math.pi/180)*np.sin(rim_lat_temp[:]*math.pi/180) - np.sin(y_center_mass_degrees*math.pi/180)*np.cos(rim_lat_temp[:]*math.pi/180)*np.cos((rim_lon_temp[:]-x_center_mass_degrees)*math.pi/180)
bearing[:]      = np.arctan2(atan2_part1[:], atan2_part2[:])*180./math.pi  #where this is -180° to +180°, clockwise, and due North is 0°

#THIS IS AN APPROXIMATION and ideally should actually trace Great Circles from
# the center of mass to the original rim points.  Instead, this just uses the
# sine and cosine components of the bearing from the center of mass to re
# -project the rim points.
rim_lon_temp[:] = dist[:] * np.sin(np.multiply(bearing[:],math.pi/180.))
rim_lat_temp[:] = dist[:] * np.cos(np.multiply(bearing[:],math.pi/180.))

ctr_lon, ctr_lat = rim_lon_temp.mean(), rim_lat_temp.mean()
# compute angle
rho, phi = cart2pol(rim_lon_temp - ctr_lon, rim_lat_temp - ctr_lat)

plt.subplot(1, 2, 1)
plt.plot(rim_lon_temp, rim_lat_temp)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
idx = phi.argsort()
phi = phi[idx]
rho = rho[idx]
plt.subplot(1, 2, 2)
plt.plot(phi / np.pi * 180, rho, 'x ', ms=2)
plt.xlabel('angle')
plt.ylabel('distance')
plt.savefig('polar.pdf', bbox_inches='tight')
plt.close()

chi2_best = 1e300

def piecewise_model(phi, phi_cp, rho_cp):
    idx = np.argsort(phi_cp)
    phi_cp = phi_cp[idx]
    rho_cp = rho_cp[idx]
    
    # make a linear interpolation between the change points
    # for those between the last and first change points:
    mask1 = phi < phi_cp[0]
    mask2 = phi > phi_cp[-1]
    #print(phi_cp, rho_cp)
    mask = np.logical_or(mask1, mask2)
    rho_predict = np.ones_like(phi) * np.nan
    rho_predict[~mask] = np.interp(phi[~mask], phi_cp, rho_cp)
    
    rho_predict[mask1] = np.interp(phi[mask1] + 2*pi, [phi_cp[-1], phi_cp[0] + 2 * pi], [rho_cp[-1], rho_cp[0]])
    rho_predict[mask2] = np.interp(phi[mask2], [phi_cp[-1], phi_cp[0] + 2 * pi], [rho_cp[-1], rho_cp[0]])
    
    return rho_predict

@mem.cache
def minfunc(params, plot=False):
    #print('params:', params)
    num_change_points = len(params) // 2
    phi_cp = np.fmod(np.asarray(params[:num_change_points]) + 3*pi, 2*pi) - pi
    if (phi_cp < -pi).any() or (phi_cp > pi).any():
        return 1e200
    rho_cp = np.asarray(params[num_change_points:])
    rho_predict = piecewise_model(phi, phi_cp, rho_cp)
    
    # compute deviance
    chi2 = np.sum((rho_predict - rho)**2)
    """
    if plot:
        global chi2_best
        if chi2 < chi2_best:
            if chi2 < chi2_best - 1.0 and plot:
                plt.plot(phi_cp / np.pi * 180, rho_cp, 'x ', color='k')
                plt.plot(phi / np.pi * 180, rho, 'x ', ms=2)
                plt.plot(phi / np.pi * 180, rho_predict, '.')
                plt.xlabel('angle')
                plt.ylabel('distance')
                plt.savefig('polar-minfunc.pdf', bbox_inches='tight')
                plt.close()
                chi2_best = chi2
            elif not plot:
                chi2_best = chi2
            #print("chi2: %.1f" % chi2, params)
    """
    return chi2

@mem.cache
def minimize(x0):
    return scipy.optimize.minimize(minfunc, x0=x0)


@mem.cache
def minimize_otherdata(x0, rho, phi):
    def minfunc_otherdata(params, plot=False):
        num_change_points = len(params) // 2
        phi_cp = np.fmod(np.asarray(params[:num_change_points]) + 3*pi, 2*pi) - pi
        if (phi_cp < -pi).any() or (phi_cp > pi).any():
            return 1e200
        rho_cp = np.asarray(params[num_change_points:])
        rho_predict = piecewise_model(phi, phi_cp, rho_cp)
        
        # compute deviance
        chi2 = np.sum((rho_predict - rho)**2)
        return chi2

    return scipy.optimize.minimize(minfunc_otherdata, x0=x0)

@mem.cache
def k_fold_validate(K, params):
    """
    Return mean chi^2 of predictions 
    when leaving out one K-fold piece at a time
    """
    num_change_points = len(params) // 2
    bin = ((phi + pi) * K / (2 * pi)).astype(int)
    chi2s = []
    variances = []
    for i in range(10):
        phi_train = phi[bin == i]
        rho_train = phi[bin == i]
        phi_test = phi[bin != i]
        rho_test = phi[bin != i]
        results = minimize_otherdata(params, rho_train, phi_train)
        
        phi_cp = np.fmod(np.asarray(results.x[:num_change_points]) + 3*pi, 2*pi) - pi
        if (phi_cp < -pi).any() or (phi_cp > pi).any():
            return 1e200
        rho_cp = np.asarray(results.x[num_change_points:])
        rho_predict = piecewise_model(phi_test, phi_cp, rho_cp)
        chi2 = ((rho_predict - rho_test)**2).sum()
        chi2s.append(chi2)
        variances.append(np.var(rho_predict))
    
    return np.mean(chi2s), np.mean(variances)

@mem.cache
def bootstrap_validate(params, nrounds=20, npatches=20):
    """
    Return mean chi^2 of predictions 
    when leaving out one some fraction piece at a time
    """
    num_change_points = len(params) // 2
    bin = ((phi + pi) * npatches / (2 * pi)).astype(int)
    chi2s = []
    variances = []
    for i in range(nrounds):
        mask = np.isin(bin, np.random.choice(npatches, size=npatches))
        if mask.all(): continue
        phi_train = phi[mask]
        rho_train = phi[mask]
        phi_test = phi[~mask]
        rho_test = phi[~mask]
        results = minimize_otherdata(params, rho_train, phi_train)
        
        phi_cp = np.fmod(np.asarray(results.x[:num_change_points]) + 3*pi, 2*pi) - pi
        if (phi_cp < -pi).any() or (phi_cp > pi).any():
            return 1e200
        rho_cp = np.asarray(results.x[num_change_points:])
        rho_predict = piecewise_model(phi_test, phi_cp, rho_cp)
        chi2 = ((rho_predict - rho_test)**2).sum()
        chi2s.append(chi2)
        variances.append(np.var(rho_predict))
    
    return np.mean(chi2s), np.mean(variances)



num_change_points = 12
sequence = []
np.random.seed(3)

best_chi2 = 1e300
best_params = None

for i in range(6):
    initial_cp_phi = np.linspace(-pi, pi, num_change_points+1)[:-1] + i / 20. * 2 * pi / num_change_points
    initial_cp_rho = np.interp(initial_cp_phi, phi, rho)
    params = list(initial_cp_phi) + list(initial_cp_rho)
    minfunc(params)
    results = minimize(x0=params) #, method='Nelder-Mead')
    chi2_best = 1e300
    chi2 = minfunc(results.x, plot=False)
    print("best fit:", chi2)
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_params = results.x

# go through each cp and try to remove it

chi2, params = best_chi2, best_params

while num_change_points > 2:
    best_chi2 = 1e300
    best_params = None
    
    # do a k-fold validation and compute the mean error
    # the choice of K here can influence the result though, because
    # it determines the prediction angle
    #test_mean, test_variance = k_fold_validate(K=num_change_points * 2, params=params)
    test_mean, test_variance = bootstrap_validate(params=params, npatches=40, nrounds=40)
    
    removal_sensitivity = []

    for i in range(num_change_points):
        print("removing point %d/%d" % (i, num_change_points))
        initial_cp_phi = np.fmod(np.asarray(params[:num_change_points]) + 3*pi, 2*pi) - pi
        initial_cp_rho = np.asarray(params[num_change_points:])
        
        mask = np.ones(num_change_points, dtype=bool)
        mask[i] = False
        initial_cp_phi = initial_cp_phi[mask]
        initial_cp_rho = initial_cp_rho[mask]


        start_params = list(initial_cp_phi) + list(initial_cp_rho)
        assert len(start_params) == len(params) - 2, (len(start_params), len(params), len(mask), mask.sum())
        result_chi2 = minfunc(start_params)
        #removal_sensitivity.append(result_chi2)
        results = minimize(x0=start_params) #, method='Nelder-Mead')
        chi2_best = 1e300
        result_chi2 = minfunc(results.x, plot=False)
        removal_sensitivity.append(result_chi2)
        if result_chi2 < best_chi2:
            best_chi2 = result_chi2
            best_params = results.x
    
    sequence.append([num_change_points, chi2, params, test_mean, test_variance, np.mean(removal_sensitivity), np.std(removal_sensitivity)])
    chi2, params = best_chi2, best_params
    num_change_points = num_change_points - 1
    assert len(params) == 2 * num_change_points, (params, num_change_points)

plt.subplot(1, 2, 1)
plt.plot(rim_lon_temp, rim_lat_temp, 'x ', ms=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#f_sigma = 0.009

plt.subplot(1, 2, 2)
#aics = []
plt.plot(phi / np.pi * 180, rho, 'x ', ms=2)

for num_change_points, chi2, params, _, _, _, _ in sequence:
    phi_cp = np.fmod(np.asarray(params[:num_change_points]) + 3*pi, 2*pi) - pi
    rho_cp = np.asarray(params[num_change_points:])
    rho_predict = piecewise_model(phi, phi_cp, rho_cp)
    f_sigma = (((rho_predict - rho) / rho_predict)**2).mean()**0.5
    
    #aic = chi2 / ((f_sigma * rho_predict)**2).sum() + 2 * len(params)
    #aics.append(aic)
    print('%2d changepoints: chi2=%.1f' % (num_change_points, chi2))
    plt.subplot(1, 2, 2)
    plt.plot(phi / np.pi * 180, rho_predict, '-', lw=1, label='%d' % num_change_points)
    plt.subplot(1, 2, 1)
    lon_predict, lat_predict = pol2cart(rho_predict, phi)
    lon_predict, lat_predict = lon_predict + ctr_lon, lat_predict + ctr_lat
    plt.plot(lon_predict, lat_predict, lw=1, label='%d' % num_change_points)

plt.subplot(1, 2, 1)
plt.legend(loc='best')
plt.subplot(1, 2, 2)
plt.legend(loc='best')
plt.xlabel('angle')
plt.ylabel('distance')
plt.savefig('polar.pdf', bbox_inches='tight')
plt.close()

num_change_points, chi2, params, tavg, tstd, avg, std = zip(*sequence)

# bias is chi2 of prediction (tavg)
# variance is variance of prediction minus square of mean prediction (tstd).

plt.plot(num_change_points, chi2, label='best fit chi2')
plt.plot(num_change_points, avg, label='CP removal: chi2 average')
plt.plot(num_change_points, std, label='CP removal: chi2 variance')
plt.plot(num_change_points, np.asarray(tavg), label='bias')
plt.plot(num_change_points, np.asarray(tstd), label='variance')
plt.plot(num_change_points, (np.asarray(tstd) + np.asarray(tavg)), label='bias+variance')
best_ncp = np.argmin(np.asarray(tstd) + np.asarray(tavg))
plt.plot(num_change_points[best_ncp], (np.asarray(tstd) + np.asarray(tavg))[best_ncp], 'o')
print('best number of change points: %d' % num_change_points[best_ncp])
#plt.plot(num_change_points, np.asarray(aics) * 100, label='AIC')
plt.yscale('log')
plt.xlabel('number of change points')
plt.ylabel('chi2')
plt.legend(loc='best')
plt.savefig('polar-biasvar.pdf', bbox_inches='tight')
plt.close()



