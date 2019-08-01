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

#Set any code-wide parameters.
np.set_printoptions(threshold=sys.maxsize) #for debugging purposes


#General help information.
parser = argparse.ArgumentParser(description='Identify polygonal crater rims.')

#Various runtime options.
parser.add_argument('--input',                              dest='inputFile',                       action='store', default='',     help='Name of the CSV file with a single crater rim trace with latitude,longitude in decimal degrees, one per line.')
parser.add_argument('--body_radius',                        dest='d_planet_radius',                 action='store', default='1',    help='Radius of the planetary body on which the crater is placed, in kilometers.')
parser.add_argument('--tolerance_distance_min_forside',     dest='tolerance_distance_min_forside',  action='store', default='5',    help='The minimum length of a rim segment to be approximately straight to be considered an edge (in km).')
parser.add_argument('--tolerance_distance_max_forhinge',    dest='tolerance_distance_max_forhinge', action='store', default='5',    help='The maximum length of a rim segment to be curved enough to be considered a hinge/joint between edges (in km).')
parser.add_argument('--tolerance_angle_max_forside',        dest='tolerance_angle_max_forside',     action='store', default='10',   help='The maximum angle that a rim segment can vary for it to be considered a straight side (in degrees).')
parser.add_argument('--tolerance_angle_min_forhinge',       dest='tolerance_angle_min_forhinge',    action='store', default='20',   help='The minimum standard deviation of the bearing angle a rim segment must vary within the given length for it to be considered a hinge/joint between edges (in degrees).')
parser.add_argument('--smoothing_length',                   dest='smoothing_length',                action='store', default='1',    help='The rim trace will be smoothed using a Savitzky-Golay filter of a length in rim points where that length is divided by THIS value (e.g., 1 means the smoothing length is the average number of points corresponding to the minimum length of an edge).')

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



##--------- CALCULATE DISTANCE AND BEARING VECTORS OF THE RIM TRACE ----------##

#Re-purpose the "dist" vector to make it equal to the distance between each
# point as you walk around the polygon.
dist[0:-1]  = np.power((np.sin((rim_data[1:,1]-rim_data[0:-1,1])*math.pi/180/2)),2) + np.cos(rim_data[1:,1]*math.pi/180)*np.cos(rim_data[0:-1,1]*math.pi/180)*np.power(np.sin((rim_data[1:,0]-rim_data[0:-1,0])*math.pi/180/2),2)
dist[-1]    = np.power((np.sin((rim_data[0 ,1]-rim_data[-1  ,1])*math.pi/180/2)),2) + np.cos(rim_data[0 ,1]*math.pi/180)*np.cos(rim_data[-1  ,1]*math.pi/180)*np.power(np.sin((rim_data[0  ,0]-rim_data[-1 ,0])*math.pi/180/2),2)
dist[:]     = 2*np.arctan2(np.sqrt(dist[:]),np.sqrt(1.0 + np.negative(dist[:]))) * float(args.d_planet_radius)


#Now, to be fair, I lied.  About all the above stuff.  What we really want to do
# is use a *smoothed* rim trace because, otherwise, bumps and wiggles will throw
# things off and result in really short sides, which is something the eye does
# not do.  We needed to calculate things up to here in order to get the
# smoothing length, which is based on the average spacing relative to the
# tolerance_distance_min_forside length scale.

#Determine the smoothing distance.  This is based on what worked.  The first
# line calculates the average number of rim points per minimum length of an edge
# and what we want is the second line, which is for an edge to be some number of
# points, at a minimum.  But, for the Savitzky-Golay filter, we must have an odd
# number of points to smooth over, hence the third line.
smoothing_distance = float(args.tolerance_distance_min_forside) / np.mean(dist)
smoothing_distance = int(smoothing_distance/float(args.smoothing_length))
smoothing_distance = smoothing_distance+1 if (smoothing_distance % 2 == 0) else smoothing_distance
print("Smoothing distance in units of rim points:", smoothing_distance)

#Perform the smoothing.
rim_data_backup = rim_data.copy()
rim_data_lon = sp.signal.savgol_filter(rim_data[:,0],smoothing_distance,3, mode='wrap')
rim_data_lat = sp.signal.savgol_filter(rim_data[:,1],smoothing_distance,3, mode='wrap')
rim_data[:,0] = rim_data_lon.copy()
rim_data[:,1] = rim_data_lat.copy()


#Now, redo the distance and then continue on.
dist[0:-1]  = np.power((np.sin((rim_data[1:,1]-rim_data[0:-1,1])*math.pi/180/2)),2) + np.cos(rim_data[1:,1]*math.pi/180)*np.cos(rim_data[0:-1,1]*math.pi/180)*np.power(np.sin((rim_data[1:,0]-rim_data[0:-1,0])*math.pi/180/2),2)
dist[-1]    = np.power((np.sin((rim_data[0 ,1]-rim_data[-1  ,1])*math.pi/180/2)),2) + np.cos(rim_data[0 ,1]*math.pi/180)*np.cos(rim_data[-1  ,1]*math.pi/180)*np.power(np.sin((rim_data[0  ,0]-rim_data[-1 ,0])*math.pi/180/2),2)
dist[:]     = 2*np.arctan2(np.sqrt(dist[:]),np.sqrt(1.0 + np.negative(dist[:]))) * float(args.d_planet_radius)


#Integrate the distances to put the sum at the NEXT point, so point index 1 has
# the distance between point index 0 and 1.
#TO DO: Python-ize.
dist_INT = [0]*(len(rim_data)+1)
for iCounter in range(1,len(dist_INT)):
    dist_INT[iCounter] = dist_INT[iCounter-1] + dist[iCounter-1]

#Re-purpose the "bearing" vector to make it equal to the bearing between each
# point as you walk around the polygon.
atan2_part1[0:-1] = np.sin((rim_data[1:,0]-rim_data[0:-1,0])*math.pi/180)*np.cos(rim_data[1:,1]*math.pi/180)
atan2_part1[  -1] = np.sin((rim_data[0 ,0]-rim_data[  -1,0])*math.pi/180)*np.cos(rim_data[0 ,1]*math.pi/180)
atan2_part2[0:-1] = np.cos(rim_data[0:-1,1]*math.pi/180)*np.sin(rim_data[1:,1]*math.pi/180) - np.sin(rim_data[0:-1,1]*math.pi/180)*np.cos(rim_data[1:,1]*math.pi/180)*np.cos((rim_data[1:,0]-rim_data[0:-1,0])*math.pi/180)
atan2_part2[  -1] = np.cos(rim_data[  -1,1]*math.pi/180)*np.sin(rim_data[0 ,1]*math.pi/180) - np.sin(rim_data[  -1,1]*math.pi/180)*np.cos(rim_data[0 ,1]*math.pi/180)*np.cos((rim_data[0 ,0]-rim_data[  -1,0])*math.pi/180)
bearing           = np.arctan2(atan2_part1, atan2_part2)*180./math.pi   #where this is -180° to +180°, clockwise, and due North is 0°

#Make the bearing continuous when we go over +180°.
#TO DO: Python-ize.
for iCounter in range(1,len(bearing)):
    bearing[iCounter] = bearing[iCounter]+360. if np.abs(bearing[iCounter]-bearing[iCounter-1])>180. else bearing[iCounter]

#But, what we actually want is the DIFFERENCE in bearing from one point to the
# next, but not quite, we want this point relative to surrounding ones, so we
# will differentiate using the central difference method.
#TO DO: Python-ize.
bearing_DIF = [0]*(len(rim_data))
bearing_DIF[0] = bearing[1]-bearing[0]
bearing_DIF[len(bearing_DIF)-1] = bearing[len(bearing_DIF)-1]-bearing[len(bearing_DIF)-2]
for iCounter in range(1,len(bearing_DIF)-2):
    bearing_DIF[iCounter] = ( (bearing[iCounter+1]-bearing[iCounter]) + (bearing[iCounter]-bearing[iCounter-1]) ) / 2.



##----------------------- PERFORM THE POLYGON ANALYSIS -----------------------##

#Vectors and Variables.
array_sides  = []       #stores as a tuple the indices of the start and end of any edge
array_angles = []       #stores the average bearing of any found edge
array_length = []       #stores the length along the rim for any found edge
counter_point_start = 0

#Large loop to do the math.  This loop will walk around the crater rim and,
# based on the four command-line argument tolerances, will determine if and
# where there are any polygonal edges and/or hinges.  Rather than explain how it
# works up here, I will walk you through it as we go.  To start off with, we are
# going to loop through each point along the rim and determine if there are any
# edges or hinges from each point.  But, if we find one, we skip to the end of
# it to determine any remaining, such that this is set up as a while-True loop
# instead of for() loop because Python does not allow you to dynamically alter
# the iterating variable within the loop itself.
while True:

    #For the initial search from this starting point, we need the index of the
    # first possible end point for this edge, which is based on distance.
    counter_point_end = round(np.searchsorted(dist_INT, dist_INT[counter_point_start]+float(args.tolerance_distance_min_forside)))

    #NumPY will NOT return an error if the search is before or after the list,
    # so we need to check for that.
    if (counter_point_end > 0) and (counter_point_end < len(dist_INT)):

        #We have a set of points that could be an edge because it's long enough.
        # The first step in testing it is to calculate the standard deviation.
        # In this calculation, we want the end points to be inclusive, so we
        # need to slice up to +1.  We also want the sample standard deviation,
        # not the population standard deviation, so need to use ddof=1.
        standardDeviation = np.std(bearing[counter_point_start:counter_point_end+1], ddof=1)

        #Now, test that standard deviation.
        if standardDeviation <= float(args.tolerance_angle_max_forside):

            #We successfully found points that can be considered a side, so now
            # want to look further along the rim to determine if any more
            # contiguous points could be considered part of this side, too.
            while True:
                counter_point_end += 1
                standardDeviation = np.std(bearing[counter_point_start:counter_point_end+1], ddof=1)
                if (standardDeviation > float(args.tolerance_angle_max_forside)) or (counter_point_end >= len(rim_data)-2):
                    counter_point_end -= 1  #subtract 1 because we went over
                    break

            #We have our maximum-length rim section that qualifies as an edge,
            # so now re-calculate the standard deviation of the bearing of the
            # points within it.
            reference_standardDeviation = np.std(bearing[counter_point_start:counter_point_end+1], ddof=1)

            #See if shifting the edge back-and-forth at all allows it to be
            # extended or shifted to better represent an edge.
            shift_start = +1
            shift_end   = +1 if counter_point_end < len(rim_data)-2 else 0
            reference_length = dist_INT[counter_point_end+1]-dist_INT[counter_point_start]
            while True:

                #Start off by shifting 1 point along the rim.
                counter_point_start_test = counter_point_start + shift_start
                counter_point_end_test   = counter_point_end   + shift_end

                #Ensure the length is still long enough for our threshold; if
                # not, increase the end point until it is.
                if dist_INT[counter_point_end_test+1]-dist_INT[counter_point_start_test] < float(args.tolerance_distance_min_forside):
                    while True:
                        shift_end += 1
                        counter_point_end_test = counter_point_end + shift_end
                        if dist_INT[counter_point_end_test+1]-dist_INT[counter_point_start_test] >= float(args.tolerance_distance_min_forside):
                            break   #break if it's long enough
                        if counter_point_end + shift_end >= len(rim_data):
                            break   #break if we go over the edge -- TO DO: Make wrap-around aware.

                #Now check to see if the standard deviation of the bearings both
                # meets our requirements for a maximum to still be a side, and
                # is better than the original side we found.  If it is, then set
                # new reference values and set the shifts to go further.  If not
                # then decrease the shift back to the previous loop, and break.
                standardDeviation = np.std(bearing[counter_point_start_test:counter_point_end_test+1], ddof=1)
                if (standardDeviation <= float(args.tolerance_angle_max_forside)) and (standardDeviation < reference_standardDeviation):
                    shift_start += 1
                    shift_end   += 1 if counter_point_end + shift_end < len(rim_data)-2 else 0
                    reference_length = dist_INT[counter_point_end_test+1]-dist_INT[counter_point_start_test]
                    reference_standardDeviation = np.std(bearing[counter_point_start_test:counter_point_end_test+1], ddof=1)
                else:
                    shift_start -= 1
                    shift_end   -= 1
                    break

            #Set the start/end points to the results from above.
            counter_point_start += shift_start
            counter_point_end   += shift_end

            #Determine if we can correct back at all, possibly extending either
            # start or end by one point.  We still need to check angular varia-
            # tion, but since we are EXTENDING the sides, we don't need to check
            # for length.
            flag_start_decrease = 0
            flag_stop_increase  = 0
            if (counter_point_start >= 1) and (counter_point_start > (array_sides[len(array_sides)-1][1] if len(array_sides)>0 else 0)):    #TO DO: Make wrap-around-aware
                standardDeviation = np.std(bearing[counter_point_start-1:counter_point_end+1], ddof=1)
                if standardDeviation <= float(args.tolerance_angle_max_forside):
                    flag_start_decrease -= 1
                counter_point_start += flag_start_decrease
            if counter_point_end+2 <= len(rim_data):    #TO DO: Make wrap-around-aware
                standardDeviation = np.std(bearing[counter_point_start:counter_point_end+2], ddof=1)
                if standardDeviation <= float(args.tolerance_angle_max_forside):
                    flag_stop_increase  += 1
                counter_point_end   += flag_stop_increase
            reference_standardDeviation = np.std(bearing[counter_point_start:counter_point_end+1], ddof=1)
            reference_length = dist_INT[counter_point_end+1]-dist_INT[counter_point_start]

            #Now that we have the for-realz edge start/end indices, store them.
            array_sides.append([counter_point_start,counter_point_end+1])

            #Since we have a real edge, store the average bearing of this side.
            array_angles.append(np.mean(bearing[counter_point_start:counter_point_end+1]))

            #Since we have a real edge, store the length along the rim for it.
            array_length.append(dist_INT[counter_point_end+1]-dist_INT[counter_point_start])

            #Set up to testfor another edge at the end of this one.
            counter_point_start = counter_point_end+1

        #The standard deviation of the minimum-length side was too large, so it
        # does not count as an edge and we have to move on, starting with the
        # next point as the possible start location.
        else:
            counter_point_start += 1

    #The distance measure for an edge failed to find something within the list,
    # so we should move on.
    #TO DO: This is where wrap-around code needs to be developed.
    else:
        counter_point_start += 1

    #Our only, singular quit criterion.
    if counter_point_start > len(dist)-1:
        break



#Now determine if the possible hinges meet the criteria set by the command-line
# arguments for maximum distance and minimum angle.  Special case for the last
# candidate hinge to support wrap-around.
array_hinge_valid = [0]*(len(array_angles)) #holds a boolean array
if len(array_angles) > 0:

    #All angles but the last edge to first edge.
    for counter_hinge in range(0,len(array_angles)-1):
        if(dist_INT[array_sides[counter_hinge+1][0]]-dist_INT[array_sides[counter_hinge][1]] < float(args.tolerance_distance_max_forhinge)):
            if(array_angles[counter_hinge+1]-array_angles[counter_hinge] > float(args.tolerance_angle_min_forhinge)):
                array_hinge_valid[counter_hinge] = 1

    #Wrap-around for the last angle.
    if(dist_INT[array_sides[0][0]]+(dist_INT[len(dist_INT)-1]-dist_INT[array_sides[len(array_angles)-1][1]]) < float(args.tolerance_distance_max_forhinge)):
        if((array_angles[0]+360)-array_angles[len(array_angles)-1] > float(args.tolerance_angle_min_forhinge)):
            array_hinge_valid[len(array_hinge_valid)-1] = 1



##Debug purposes.
#print(array_sides)
#print(array_angles)
#print(array_length)
#print(array_hinge_valid)



##------------------------ OUTPUT RESULTS TO THE USER ------------------------##

print("\nThere were %g edges and %g hinges found.  Data follows for each.\n" % (len(array_sides), np.sum(array_hinge_valid)))
for iCounter, edge in enumerate(array_sides):
    print(" Edge #%g" % int(iCounter+1))
    print("   Start  (Latitude, Longitude)      :", round(rim_data[edge[0],0],3), round(rim_data[edge[0],1],3))
    print("   End    (Latitude, Longitude)      :", round(rim_data[edge[1],0],3), round(rim_data[edge[1],1],3))
    print("   Center (Latitude, Longitude)      :", round(np.mean(rim_data[array_sides[iCounter][0]:array_sides[iCounter][1],0]),3), round(np.mean(rim_data[array_sides[iCounter][0]:array_sides[iCounter][1],1]),3))
    print("   Length (km)                       :", round(array_length[iCounter],3))
    print("   Bearing (degrees, N=0°, CW, w/ SD):", round(array_angles[iCounter],3), round(np.std(bearing[edge[0]:edge[1]], ddof=1),3))
print("")
for iCounter, hinge in enumerate(array_hinge_valid):
    if hinge == 1:
        if iCounter < len(array_hinge_valid)-1:
            print(" Candidate Hinge #%g" % int(iCounter+1))
            print("   Center (Latitude, Longitude)      :", round((rim_data[int(round((array_sides[iCounter+1][0] + array_sides[iCounter][1]) / 2.))][0]),3), round((rim_data[int(round((array_sides[iCounter+1][0] + array_sides[iCounter][1]) / 2.))][1]),3))
            print("   Length (km)                       :", round(dist_INT[array_sides[iCounter+1][0]]-dist_INT[array_sides[iCounter][1]],3))
            print("   Angle  (degrees, N=0°, CW)        :", round(array_angles[iCounter+1]-array_angles[iCounter],3))
        else:
            print(" Candidate Hinge #%g" % int(iCounter+1))
            print("   Center (Latitude, Longitude)      :", round((rim_data[int(round((array_sides[0][0] + array_sides[iCounter][1]) / 2.))][0]),3), round((rim_data[int(round((array_sides[0][0] + array_sides[iCounter][1]) / 2.))][1]),3))
            print("   Length (km)                       :", round(dist_INT[array_sides[0][0]]+(dist_INT[len(dist_INT)-1]-dist_INT[array_sides[iCounter][1]]),3))
            print("   Angle  (degrees, N=0°, CW)        :", round(array_angles[0]+360-array_angles[iCounter],3))
    else:
        print(" Candidate Hinge #%g failed to meet tolerances." % int(iCounter+1))

#Output the time to the user (do it now because the display, below, will not end
# the code so the time will be however long the graph stays open.
print("\nTime to analyze the crater: %f sec." % round((time.time()-timer_start),5))




##--------------------------------- DISPLAY ----------------------------------##

#I'm going to comment this as though you do not know anything about Python's
# graphing capabilities with MatPlotLib.  Because I don't know anything about it

#Create the plot reference.
PolygonalCraterWindow = mpl.figure(1, figsize=(10,10))


#Plot the smoothed crater rim.
mpl.plot(rim_data[:,0], rim_data[:,1], color='#AAAAAA', linewidth=7, label='Smoothed Rim Trace')

#Plot the crater rim.
mpl.plot(rim_data_backup[:,0], rim_data_backup[:,1], color='#666666', linewidth=3, label='Rim Trace')


#Append a line for every valid edge.
for iCounter in range(len(array_sides)):

    #For the idealized edges, we can't simply take the start point and graph to
    # the end point.  Instead, we take the center lat/lon of each edge and
    # extend it outwards to the edge end, assuming simple linear projection.
    #*****WARNING*****: This will not be accurate on small bodies!!
    #TO DO: Take the average center and trace out Great Circles for the edges
    # instead of just drawing a straight line.
    center_x = np.mean(rim_data[array_sides[iCounter][0]:array_sides[iCounter][1],0])
    center_y = np.mean(rim_data[array_sides[iCounter][0]:array_sides[iCounter][1],1])
    length_x = abs(dist_INT[array_sides[iCounter][0]] - dist_INT[array_sides[iCounter][1]]) * np.sin(array_angles[iCounter]*math.pi/180.)
    length_y = abs(dist_INT[array_sides[iCounter][0]] - dist_INT[array_sides[iCounter][1]]) * np.cos(array_angles[iCounter]*math.pi/180.)
    length_x *= 360./(2.*float(args.d_planet_radius)*math.pi)
    length_y *= 360./(2.*float(args.d_planet_radius)*math.pi)

    linesegment_x = [center_x-length_x/2., center_x+length_x/2.]
    linesegment_y = [center_y-length_y/2., center_y+length_y/2.]
    linesegment_x_extended = [center_x-length_x/2.*2.0, center_x+length_x/2.*2.0]
    linesegment_y_extended = [center_y-length_y/2.*2.0, center_y+length_y/2.*2.0]

    #Append the lines for this edge.
    if iCounter == 0:
        mpl.plot(linesegment_x_extended, linesegment_y_extended, color='#FFAAAA', linewidth=1, dashes=[5,5], label='Edges Extended')
        mpl.plot(linesegment_x, linesegment_y, color='#FF0000', linewidth=2, label='Edges')
    else:
        mpl.plot(linesegment_x_extended, linesegment_y_extended, color='#FFAAAA', linewidth=1, dashes=[5,5])
        mpl.plot(linesegment_x, linesegment_y, color='#FF0000', linewidth=2)

#Append a symbol for every valid hinge.
array_hinges_x_position = []
array_hinges_y_position = []
for iCounter, validHinge in enumerate(array_hinge_valid):
    if validHinge == 1:
        if iCounter < len(array_hinge_valid)-1:
            array_hinges_x_position.append([(rim_data[int(round((array_sides[iCounter+1][0] + array_sides[iCounter][1]) / 2.))][0])])
            array_hinges_y_position.append([(rim_data[int(round((array_sides[iCounter+1][0] + array_sides[iCounter][1]) / 2.))][1])])
        else:
            array_hinges_x_position.append([(rim_data[int(array_sides[0][0])][0] + rim_data[int(array_sides[iCounter][1])][0]) / 2.])
            array_hinges_y_position.append([(rim_data[int(array_sides[0][0])][1] + rim_data[int(array_sides[iCounter][1])][1]) / 2.])
mpl.scatter(array_hinges_x_position, array_hinges_y_position, s=250, facecolors='none', edgecolors='#0000FF', label='Hinges')


##General graph appendages.

#Append the legend to the plot.
mpl.legend(loc='upper right')

#Append axes labels.
mpl.xlabel('Longitude (degrees)')
mpl.ylabel('Latitude (degrees)')

#Append graph title.
mpl.title('Crater Rim with Any Polygonal Edges and Angles')

#Make axes equal / square in degrees space.
mpl.axis('equal')


##Finally, make the plot visible.
mpl.show()
