# COPYRIGHT NOTICE:  Portions of this code were originally written by Alex H.
# Parker for this work.  The majority of this code's current version were
# written by Stuart J. Robbins <stuart@boulder.swri.edu>.  This code is licensed
# as: CC BY-NC-SA, meaning it is licensed under the Creative Commons license
# that attribution is needed, non-commercial applications are permitted, and any
# derivative work must be distributed under the same license as the original.
#
# This code makes use of geo-py-0.4, which is distributed in its own folder with
# its own license that permits redistribution provided the original license is
# maintained, and further details are available in that file.
# Original distribution of geo-py-0.4: https://github.com/gojuno/geo-py

# DEPENDENCIES:  This code requires several additional Python packages which
# can be freely installed via a normal installation process.  Those packages are
#   - pandas
#   - matplotlib
#   - scipy
#   - numpy  #not actually needed as a separate install due to its inclusion
#             with scipy
#   - shapely
# Additionally, the "geo" folder must be included in the same folder as this
# Python file.
# Example install in a clean conda environment (numpy is installed automatically
# with scipy).
#  $ conda create -n polygonal
#  $ conda activate polygonal
#  $ conda install pandas
#  $ conda install scipy
#  $ conda install matplotlib
#  $ conda install shapely

# USAGE: The code takes several command-line arguments.  Default values exist
# for all them except the first, so only the first must be explicitly specified:
#   --data      ~CSV file in units of decimal degrees, labeled with "lon" and
#                "lat" (so first row must have those columns labeled)
#   --minPoly   ~minimum n-gon to try
#   --maxPoly   ~maximum n-gon to try
#   --minArcs   ~minimum number of sides to be arcs (including 0)
#   --maxArcs   ~total number of sides to be arcs, though there is a hard-coded
#                max that will overrule this
#   --repeat    ~how many times the code will try to fit a polygon with the SAME
#                parameters, where "1" is to do it once, while larger values
#                will fit multiple times so the user can test for robustness
#   --ellipsoid ~valid values are Mercury, Venus, Earth, WGS84, Moon, Mars,
#                Vesta, Ceres; any other bodies and you will need to add support
#
# A simple circle fit (minPoly 0, maxPoly 0) will be displayed for a second and
# then disappear from the screen.  Any other kind of fit will be displayed on-
# screen with the rim trace and fit overlaid, and it will iterate.  If nothing
# changes, the code is still iterating, but it is not improving the shape.  For
# each improved iteration, the improved chi-squared value will be also output
# to the terminal display along with the iteration number and number of sides
# being tested.
#
# Several files are output from the fits.  Each will have the name of the input
# file as its root and will have a .txt extension.
#
# For the first file type, output, if any shape other than a circle was fit, the
# file will have two numbers in it separated by a hyphen and then "-chi-sq"
# appended.  This is a tab-separated values (TSV) file with the iteration number
# and the chi-squared value as the two columns.  The code is set to "append"
# data such that if the code iterates (--repeat is set to >1, or you run it
# again), each new iteration will be appended.
#
# The second file output is data for any circle-fit done by this code.  For
# example, if you opt to use --minPoly 0 --maxPoly 3 then there will be three
# fits: circle, triangle, and triangle with one curved side.  A "-circdata"
# output file will have three lines that are TSV.  The first column will be the
# central longitude of the crater, second is central latitude, and third is the
# radius (not diameter) of the fit.  In the above example, the first two lines
# will be identical because no circle-fit was done to support an arc in the
# second fit (triangle), but a circle fit was still done.  The third line will
# have the data for the arc, where that data is the circle to which the arc
# belongs.  This file is similar set to "append."
#
# The third file output is data about the edges for the crater.  If a circle fit
# was explicitly done (--minPoly 0), then there will be a "-edges-circle" file
# output with two columns, TSV for the circle fit longitude and latitude with
# the same number of datapoints as the original rim trace.  This is for plotting
# purposes in your own software.  If anything other than or in addition to a
# circle fit was done (--maxPoly 3 or larger), then a "-edges" file will be
# output with four columns.  This file is also TSV.  The first column is the
# longitude of a vertex point, the second is the latitude.  The third column is
# whether that vertex point is connected to the next via a line (value of 0) or
# an arc (value of 1).  The fourth column is the azimuth of the line going from
# that vertex point to the next one (or, if the last line/point, going to the
# first).  This is clockwise from North, and the vertex points go clockwise
# around the crater.  If an arc is indicated (1 in the third column), that arc
# follows the circle described in the "-circdata" file.  These files are set to
# "append" like the others so additional runs or iterations will add to them
# rather than replace them.
#
# The fourth output file type is "-shape_forDisplay" and includes a two-column
# TSV file (longitude, latitude) that can be used in graphing the output shape
# in your software of choice.  It is similar to the "-edges-circle" except that
# it has all fitted shapes / iterations, includes a wrap-around (the last line
# for a shape is the same as the first), a blank line separating each shape
# (output is appended for each shape, as with the other files), and any arcs
# are drawn out as opposed to lines which are not (since vertex points would
# simply be connected by a straight line in graphing software).



#Import the libraries needed for this program.
import argparse                                 #allows parsing of input options
import numpy as np                              #lots of mathematics and data-handling
import math                                     #standard python library math
import random                                   #self-explanatory?
import pandas as pd                             #using this as the dataframe for the rim
import matplotlib.pyplot as plt                 #for displaying the results
from scipy import optimize                      #used for fitting a circle
from scipy.signal import savgol_filter          #specific maths!
from shapely.geometry.polygon import Polygon    #conda install shapely
import time                                     #timing for debugging

#For doing our Great Circle corrections.
import geo
import geo.sphere as sphere
import geo.ellipsoid as ellipsoid
from geo.constants import QUARTER_PI, HALF_PI, WGS84, Mercury, Venus, Moon, Mars, Vesta, Ceres



#Attempt to calculate some meaningful chi-squared metric for the polygon shape.
# At the moment, this is an area argument where we take the difference in the
# shapes of the fit and the rim trace.  As the polgyon fit improves, then the
# difference between the rim and polygon should decrease.  It is also sort of
# like a sum-of-squares approach.
def poly_chi2(x1, y1, x2, y2, index_arc, xc, yc, r, x_shape, y_shape):

    #What we want as a metric of how good this shape represents the crater is
    # the area of the non-overlap area of each polygon.  Since, if they overlap
    # perfectly, it's a perfect fit!  Fortunately, the shapely library can do
    # this for us very efficiently, so store the Python array rim trace to a
    # shapely polygon structure.
    arr_rim = [[x1[i],y1[i]] for i in range(0,len(x1))]
    shp_rim = Polygon(arr_rim)
    
    #Do the same thing for the polygon test fit, which has already been expanded
    # to trace out any arcs.
    arr_fit = [[x_shape[i],y_shape[i]] for i in range(0,len(x_shape))]
    shp_fit = Polygon(arr_fit)

    #We need to check to make sure the fit is a valid shapely shape, otherwise
    # there will be errors.  So, do that, or set the area to be bigly.
    if shp_fit.is_valid:
        difference = shp_rim.symmetric_difference(shp_fit)
        area = difference.area
    else:
        area = 1E100

    #Return the area result.
    return area




#Test to determine if a polygon is a convex shape.  Gathered from somewhere off
# the internet, I think, this was an original contribution by AHP, re-formatted
# to be consistent with the rest of the code by SJR.
def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    TWO_PI = 2 * np.pi #defining this saves math later since it is used >1x

    try:  # needed for any bad points or direction changes

        #Get starting information.
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0

        #Check each point (the side ending there, its angle) and accumulate
        # angles.
        for ndx, newpoint in enumerate(polygon):

            # Update point coordinates and side directions, check side length.
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = np.arctan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  #repeated consecutive points
                
            #Calculate and check the normalized direction-change angle.
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += TWO_PI  #make it in half-open interval (-Pi, Pi]
            elif angle > np.pi:
                angle -= TWO_PI
            if ndx == 0:  #if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  #if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  #not both positive nor both negative
                    return False

            #Accumulate the direction-change angle.
            angle_sum += angle

        #Check that the total number of full turns is plus-or-minus 1.
        return abs(round(angle_sum / TWO_PI)) == 1

    except (ArithmeticError, TypeError, ValueError):
        return False  #any exception means not a proper convex polygon




#Plot the results.
def plotMe(x_shape0,y_shape0,xs,ys,xp0,yp0,n_sides,wait):
    x_shape0 = np.append(x_shape0, x_shape0[0]) #close the polygon
    y_shape0 = np.append(y_shape0, y_shape0[0]) #close the polygon
    plt.clf()
    plt.plot(xs, ys, 'r-')                      #draw the rim trace
    plt.plot(x_shape0, y_shape0, 'k-')          #draw the polygon line
    if n_sides > 0: plt.plot(xp0, yp0, 'ko')    #draw the polygon vertices
    plt.axis('scaled')                          #make the plot square
    plt.ylim(-0.8,+0.8)
    plt.xlim(-0.8,+0.8)
    plt.draw()
    plt.pause(wait)




#Save a bunch of information to an array to return to the main function to
# decide what's the best polygon representation of the crater rim.
def writeReturnArray(n_sides,n_sides_forRealz,n_arcs,best_chi,xc,yc,fit_radius,xp0,yp0,index_arc_saved):
    #Try to come up with a meaningful measure of the degrees-of-freedom.
    if n_sides > 0:
        if n_arcs == 0:
            d_DOF = 2.0*n_sides #degrees of freedom is just the {x,y} position of each vertex
        else:
            d_DOF = 2.0*n_sides_forRealz + 2.0*n_arcs + 1 #I'm sure this is wrong, but until I hear otherwise ... the +1 is for the fit center ... but it's not actually free, it's dependent on a circle-fit to all the vertexes, and also the vertexes that end the arcs are now correlated because they must be the same radial distance from the circle center ... but WHERE that arc is (which vertex) is free, so that's another DOF ... yeah, this is complicated.
    else:
        d_DOF = 3.0 #circle-fit has 3 DOF (center latitude, center longitude, diameter)
    
    #Calculate some bastardization of the Akaike Information Criterion.
    AIC = 2.0*d_DOF - 2.0 * math.log(math.exp(best_chi/d_DOF))
#    print("Akaike Information Criterion: %f\n" % (AIC))
    
    #Debugging: Output all the points in the polygon.
#    print(xp0,yp0)
#    for point in range(0,len(xp)):
#        print(str(xp[point]) + "\t" + str(yp[point]))
#    print(str(xp[0]) + "\t" + str(yp[0]))

    #Store the data for this polygon.
    returnArray_Temp = []
#    if n_sides == 0:
#        returnArray_Temp.append([xc, yc, fit_radius])
#    else:
    for iCounter in range(0,len(xp0)):
        if n_arcs == 0:
            returnArray_Temp.append([xp0[iCounter],yp0[iCounter],0])
        else:
            if iCounter in index_arc_saved:
                returnArray_Temp.append([xp0[iCounter],yp0[iCounter],1])
            else:
                returnArray_Temp.append([xp0[iCounter],yp0[iCounter],0])
    
    #Store any circle information.
    if (n_arcs > 0) or (n_sides == 0):
        returnArray_Temp.append([xc, yc, fit_radius])
    else:
        returnArray_Temp.append([])

    #Append the AIC value to our return.
    returnArray_Temp.append([AIC])

    #Send the array back.
    return(returnArray_Temp)




#Primary fitting function.  Returns an array that holds the vertices of the best
# polygon fit for the rim that it can find, along with a boolean key for any
# vertices that are arcs, a supplement part of that array to list circle values
# for any arcs (center and radius), and a final point that gives the AIC value.
def fit_poly(xs, ys, minPoly, maxPoly, minArcs, maxArcs, n_ToDo, inputFile, array_vertex_points_ToSeed):

    #Several timers for debugging or time-sink-finding purposes.
    timer_part1_total                 = 0
    timer_part1_pickvertices          = 0
    timer_part1_checkangles           = 0
    timer_part1_makeshape             = 0
    timer_part1_pickarcs              = 0
    timer_part1_fitcircle             = 0
    timer_part1_moveverticesforcircle = 0
    timer_part1_makevectorshape       = 0
    timer_part1_checkconvex           = 0
    timer_part1_areadiff              = 0
    timing_redoForSmallAngles         = 0
    timer_part1_failedConvex          = 0
    timer_part2_total                 = 0
    timer_part2_makeshape             = 0
    timer_part2_fitcircle             = 0
    timer_part2_moveverticesforcircle = 0
    timer_part2_makevectorshape       = 0
    timer_part2_checkconvex           = 0
    timer_part2_areadiff              = 0
    timer_part2_failedConvex          = 0
    f_outputTimes                     = 0 #set to "1" to output timing information

    #Placeholder for the return array.
    returnArray     = []
    array_allshapes = []

    #Variable which will make this faster if >1: To calculate area difference
    # between the rim trace and the test polygon, we need to draw the arcs (if
    # they exist).  Default "1" is to draw one point every 1°, which means there
    # will be lots of points.  Lots of points mean the area calculation takes
    # more time.  If you increase this number, you sacrifice some accuracy for
    # gaining some speed.  A test on Meteor Crater saved maybe 6% of the
    # total time going from 1 to 2°, while 1° to 5° was a 15% time savings.
    angle_fidelity = 2  #units of decimal degrees

    #Test polygons with different numbers of sides, with some of those sides
    # being arcs.
    n_side_min = minPoly
    n_side_max = maxPoly
    n_arcs_min = minArcs
    for n_sides in range(n_side_min, n_side_max+1):
        #A value of 0 is a special case to fit a circle.  A polygon cannot have
        # sides <3, so we have to skip those if we include 0.
        if (n_sides == 1) or (n_sides == 2):
            continue
        
        #Figure out maximum arcs that can be fit.
        n_arcs_max = int(math.floor(n_sides/2)) if (maxArcs < 0) else maxArcs
        
        #Iterate over those arcs for n_sides number of sides.  Because of how
        # Python loops work, we need to go to +1 to be inclusive of the end, or
        # even inclusive of the first value of the max is 0.
        for n_arcs in range(n_arcs_min,n_arcs_max+1):
            #We are testing not only for polygon sides, but also the potential
            # that some sides are better represented by circle arcs.  That means
            # the real number of polygon sides could be less.  We test for up
            # to n_sides-2 arcs, meaning that for 4 "sides" we will test for
            # 4 sides, 3 sides 1 arc, 2 sides 2 arcs.
            n_sides_forRealz = n_sides-n_arcs
            
            #Output to the user.
            print("Trying to fit a %g-sided polygon with %g arc(s) instead of straight side(s)." % (n_sides, n_arcs))

            #Special case for n_sides = 0, to just fit a circle.  Since we can
            # generally assume the rim is full (otherwise you shouldn't be using
            # this code), a regular optimization through least-squares should be
            # a pretty good fit to the circle, and we don't need to do fancy
            # stuff.
            #This code is modified from the original posted here in the SciPy
            # Cookbook, ©2015 by "various authors."
            # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
            if n_sides == 0:
                #Placeholder chi-squared value that is large relative to what we
                # expect but will store the best chi-squared for this polygon
                # test.
                best_chi    = 1E100

                #Do the fitting.
                index_arc   = [-1]
                xp          = xs.copy()
                yp          = ys.copy()
                xp0         = xs.copy()
                yp0         = ys.copy()
                def calc_R(xc, yc):
                    #Calculate the simple trigonometric distance of each point
                    # from the center (xc, yc).
                    return ((xp-xc)**2 + (yp-yc)**2)**0.5
                def f_2(c):
                    #Calculate the algebraic distance between the data points
                    # and the mean circle centered at c=(xc, yc).
                    Ri = calc_R(*c)
                    return Ri - Ri.mean()
                center_estimate = np.mean(xp), np.mean(yp)
                center_2, ier   = optimize.leastsq(f_2, center_estimate)
                xc, yc          = center_2
                Ri_2            = calc_R(*center_2)
                fit_radius      = Ri_2.mean()
                theta           = [math.atan( (yp[iCounter]-yc)/(xp[iCounter]-xc) ) *180.0/math.pi for iCounter in range(0,len(xp))]
                for iCounter in range(0,len(xp)):
                    if xp[iCounter] < xc: theta[iCounter] += 180.0
                theta.sort()    #avoid problems with the shape going back on itself leading to bad shapely geometry
                x_shape         = [fit_radius*math.cos(theta[iCounter]*math.pi/180.0)+xc for iCounter in range(0,len(theta))]
                y_shape         = [fit_radius*math.sin(theta[iCounter]*math.pi/180.0)+yc for iCounter in range(0,len(theta))]
                x_shape0        = x_shape.copy()
                y_shape0        = y_shape.copy()
                xp              = x_shape.copy()
                yp              = y_shape.copy()
                chi             = poly_chi2(xs, ys, xp, yp, index_arc, xc, yc, fit_radius, x_shape, y_shape) #4.0 * n - 2.0 * chi0
                best_chi        = chi
                print("Area minimum for circle fit:",best_chi)
                
                #Draw the plot.
                plotMe(x_shape0,y_shape0,xs,ys,xp0,yp0,n_sides,1.0)
                
                #Store the information from this iteration.
                returnArray_Temp = writeReturnArray(n_sides,n_sides_forRealz,n_arcs,best_chi,xc,yc,fit_radius,x_shape0,y_shape0,index_arc)
                returnArray.append(returnArray_Temp)


            #We are not doing a circle.
            else:
                
                #Need to do things n_ToDo times.
                for iCounter_repeats in range(0,n_ToDo):

                    #Set timer for this initial seed shape finding.
                    timer_part1_total = time.time()
                        
                    #Placeholder chi-squared value that is large relative to
                    # what we expect but will store the best chi-squared for
                    # this polygon test.
                    best_chi    = 1E100

                    #Perform an initial, random guess at the best-fit polygon.
                    # Store the best one as determined by the chi-squared
                    # criterion.
                    n_random_guesses_for_initial_shape = 1000*n_sides #how many initial, completely random polygons to try
                    print("Creating an initial guess for polygon vertices by testing %g random versions." % (n_random_guesses_for_initial_shape))
                    for iCounter_guesses in range(0, n_random_guesses_for_initial_shape):
                        #We want to make sure we are only picking from convex
                        # polygons.
                        while True:
                        
                            #We don't want picks to be truely random: We want
                            # some spacing between vertices.
                            while True:
                                #We used an algorithm before to try to find some
                                # possible vertex points. For half of these
                                # random initial shapes, try using those as
                                # seeds.
                                timer = time.time()
                                if iCounter_guesses < int(n_random_guesses_for_initial_shape/2):
                                    if array_vertex_points_ToSeed.size < n_sides:
                                        inds = array_vertex_points_ToSeed.copy()
                                        inds.append(np.random.choice(np.arange(0, xs.size, 1), (n_sides-array_vertex_points_ToSeed.size)), replace=False)
                                    else:
                                        inds_inds = np.random.choice(np.arange(0, array_vertex_points_ToSeed.size, 1), n_sides, replace=False)
                                        inds = array_vertex_points_ToSeed[inds_inds]

                                #And, for the other half, just be random.
                                else:
                                    #Pick N-sides random indices from the crater
                                    # rim to be the random vertices and sort.
                                    inds = np.random.choice(np.arange(0, xs.size, 1), n_sides, replace=False)
                                
                                #Vertex points must be in order to avoid shapes
                                # that go back on themselves.
                                inds.sort()
                                timer_part1_pickvertices += time.time()-timer
                                
                                #Figure out the angles.  The first, diffs1, uses
                                # angles from –90° to +270°, where –90°/+270° is
                                # due south and angles go counter-clockwise. The
                                # second, diffs2, wraps things to a 0° to +360°
                                # domain and so will make sure that we factor in
                                # points just slightly on opposite sides of
                                # –90°/+270°.
                                timer = time.time()
                                xc, yc = np.mean(xs), np.mean(ys)  #simple center of the RIM TRACE
                                angles = [(math.atan((ys[inds[counter]]-yc)/(xs[inds[counter]]-xc))*180.0/math.pi) for counter in range(0,len(inds))]
                                for iCounter_angle in range(0,len(angles)):
                                    if xs[inds[iCounter_angle]] < xc: angles[iCounter_angle] += 180.0
                                diffs1 = [abs(angles[counter]-angles[counter+1]) for counter in range(0,len(angles)-1)]
                                diffs1.append(abs(angles[len(angles)-1]-angles[0]))
                                for iCounter_angle in range(0,len(angles)):
                                    if angles[iCounter_angle]   < 0 : angles[iCounter_angle] = 360 + angles[iCounter_angle]
                                diffs2 = [abs(angles[counter]-angles[counter+1]) for counter in range(0,len(angles)-1)]
                                diffs2.append(abs(angles[len(angles)-1]-angles[0]))
                                if (np.amin(diffs1) > (360.0/n_sides)*0.3) and (np.amin(diffs2) > (360.0/n_sides)*0.3):
                                    timer_part1_checkangles += time.time()-timer
                                    break
                                else:
                                    timing_redoForSmallAngles += 1

                            #Jitter the locations of those random indices from
                            # the original rim to account for noise and offsets.
                            timer = time.time()
                            xp, yp = np.random.normal(xs[inds], 0.05), np.random.normal(ys[inds], 0.05)
                            polygon = np.array([xp, yp]).T #will use this for convex testing
                            timer_part1_makeshape += time.time()-timer
                            

                            #Definitions for using scipy.optimize.leastsq for
                            # fitting a circle.  I made these private so that it
                            # knows about xp and yp, which is why they are
                            # declared after those.
                            #See: https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html
                            def calc_R(xc, yc):
                                #Calculate the simple trigonometric distance of
                                # each point from the center (xc, yc).
                                return ((xp-xc)**2 + (yp-yc)**2)**0.5

                            def f_2(c):
                                #Calculate the algebraic distance between the
                                # data points and the mean circle centered at
                                # c=(xc, yc).
                                Ri = calc_R(*c)
                                return Ri - Ri.mean()
                            
                            
                            #Seed arc information since we pass it to a function
                            # regardless of whether there's an arc or not.
                            index_arc = [-1]
                            xc, yc, r = 99999, 99999, 99999
                            inds_circ = -1
                            fit_radius = 0
                            flag_failed = 0
                            if n_arcs > 0:
                                #Pick which of the sides is going to be an arc.
                                # This is slightly complicated in that we have
                                # further restrictions: no two arcs can be next
                                # to each other.  That means that we can't just
                                # select all the index points for the arc at
                                # random.
                                index_arc = []
                                index_arc.append(random.randint(0,n_sides-1)) #the arc spans from one index to another, so we need to do a -1 here on the range
                                if n_arcs > 1:
                                    timer = time.time()
                                    if n_arcs < n_sides/2.:
                                        while True: #continue making vertices until we're done
                                            index_possible_remaining = []
                                            for i in range(1,n_sides-1):
                                                if (i not in index_arc) and (i-1 not in index_arc) and (i+1 not in index_arc):         #can't have it meet at the end nor beginning
                                                    index_possible_remaining.append(i)
                                            if (n_sides-1 not in index_arc) and (0 not in index_arc) and (1 not in index_arc):         #account for wrap-around
                                                index_possible_remaining.append(0)
                                            if (n_sides-2 not in index_arc) and (n_sides-1 not in index_arc) and (0 not in index_arc): #account for wrap-around
                                                index_possible_remaining.append(n_sides-1)
                                            if len(index_possible_remaining) == 0:  #there are cases where large numbers of arcs were spaced too far apart, not letting us have as many as we want ... TBD: Fix the above and it'll run faster because we won't have a lot of failures
                                                flag_failed = 1
                                                break
                                            else:
                                                index_arc.append(random.choice(index_possible_remaining))   #pick from what's left
                                                if len(index_arc) == n_arcs:
                                                    break
                                    else: #this is a special case where arcs MUST be every-other side; we still use the first random selector to determine if it's odd- or even-numbered sides
                                        index_arc_temp = []
                                        if index_arc[0] % 2 == 0: [index_arc_temp.append(2*iCounter)   for iCounter in range(0,n_arcs)]
                                        else                    : [index_arc_temp.append(2*iCounter+1) for iCounter in range(0,n_arcs)]
                                        index_arc = index_arc_temp.copy()
                                    timer_part1_pickarcs += time.time()-timer
                                
                                #Only proceed if we got arcs that meet the
                                # above criteria.  For example, at random, it
                                # could select the first 2 arcs separated by 2
                                # sides, meaning that there is no valid spot for
                                # a 3rd arc, in which case the above fails.
                                if flag_failed == 0:
                                    #We need to now fit a circle so's to RE-DO
                                    # the vertices that will be end points to
                                    # the arc(s).  We have to do a full-fledged
                                    # circle fit using all the points.
                                    timer = time.time()
                                    center_estimate = np.mean(xp), np.mean(yp)
                                    center_2, ier   = optimize.leastsq(f_2, center_estimate)
                                    xc, yc          = center_2
                                    Ri_2            = calc_R(*center_2)
                                    fit_radius      = Ri_2.mean()
                                    timer_part1_fitcircle += time.time()-timer
                                    
                                    #Now re-do the end points of the arc by
                                    # calculating the angle they make with the
                                    # fitted center and shifting them in radius
                                    # based on the fitted radius.  We *MUST* do
                                    # this otherwise there will be discontin-
                                    # uities in the shape.
                                    timer = time.time()
                                    for iCounter_pointsToChange in range(0,n_arcs):
                                        index_vertexInOriginalShape = index_arc[iCounter_pointsToChange] #because I got lost
                                        
                                        #Do the first.
                                        theta = math.atan( (yp[index_vertexInOriginalShape]-yc)/(xp[index_vertexInOriginalShape]-xc) ) *180.0/math.pi
                                        if xp[index_vertexInOriginalShape] < xc: theta += 180.0
                                        xp[index_vertexInOriginalShape] = fit_radius*math.cos(theta*math.pi/180.0)+xc
                                        yp[index_vertexInOriginalShape] = fit_radius*math.sin(theta*math.pi/180.0)+yc
                                        
                                        #Do the second.
                                        if index_arc[iCounter_pointsToChange] == len(xp)-1:
                                            index_vertexInOriginalShape = 0 #because I got lost
                                        else:
                                            index_vertexInOriginalShape = index_arc[iCounter_pointsToChange]+1 #because I got lost
                                        theta = math.atan( (yp[index_vertexInOriginalShape]-yc)/(xp[index_vertexInOriginalShape]-xc) ) *180.0/math.pi
                                        if xp[index_vertexInOriginalShape] < xc: theta += 180.0
                                        xp[index_vertexInOriginalShape] = fit_radius*math.cos(theta*math.pi/180.0)+xc
                                        yp[index_vertexInOriginalShape] = fit_radius*math.sin(theta*math.pi/180.0)+yc
                                    timer_part1_moveverticesforcircle += time.time()-timer
                                    
                                    #And, draw the shape so that we don't have
                                    # to do it multiple times, and make sure we
                                    # get it right, once.
                                    arr_fit = []
                                    timer = time.time()
                                    if index_arc[0] == -1:
                                        arr_fit = [[xp[i],yp[i]] for i in range(0,len(xp))]
                                    else:
                                        for i in range(0,len(xp)):
                                            if (i in index_arc):
                                                arr_fit.append([xp[i],yp[i]])
                                                theta_min     = math.ceil (math.atan( (yp[i  ]-yc)/(xp[i  ]-xc) ) *180.0/math.pi) #ceil so we don't have self-intersections in the geometry by over-shooting
                                                if xp[i] < xc: theta_min += 180.0
                                                if i == len(xp)-1:
                                                    theta_max = math.floor(math.atan( (yp[0  ]-yc)/(xp[0  ]-xc) ) *180.0/math.pi) #floor so we don't have self-intersections in the geometry by over-shooting
                                                    if xp[0] < xc: theta_max += 180
                                                else:
                                                    theta_max = math.floor(math.atan( (yp[i+1]-yc)/(xp[i+1]-xc) ) *180.0/math.pi) #floor so we don't have self-intersections in the geometry by over-shooting
                                                    if xp[i+1] < xc: theta_max += 180
                                                if theta_max > theta_min: theta_max -= 360  #we need to go CLOCKWISE, and all these codes go counter-clockwise
                                                theta_max = int(theta_max-1)
                                                theta_min = int(theta_min)
                                                if theta_min < theta_max:
                                                    for theta in range(theta_min+1,theta_max-1,angle_fidelity):
                                                        arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                                else:
                                                    for theta in range(theta_min-1,theta_max+1,-angle_fidelity):
                                                        arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                            else:
                                                arr_fit.append([xp[i],yp[i]])
                                    x_shape = [arr_fit[x][0] for x in range(0,len(arr_fit))]
                                    y_shape = [arr_fit[x][1] for x in range(0,len(arr_fit))]
                                    timer_part1_makevectorshape += time.time()-timer

                                    #Re-store the polygon used to check for
                                    # convex-ness.  We don't need to check the
                                    # entire shape with arcs because arcs will,
                                    # by definition, be convex, and it will slow
                                    # the convex test check way down.
                                    polygon = np.array([xp, yp]).T
                            else:
                                x_shape = xp.copy()
                                y_shape = yp.copy()
                            
                            #Ensure the test shape is convex -- a requirement of
                            # this problem.
                            timer = time.time()
                            if is_convex_polygon(polygon) and (flag_failed == 0):
                                timer_part1_checkconvex += time.time()-timer
                                
                                #Calculate the "chi-squared" value of this test
                                # polygon.
                                timer = time.time()
                                chi = poly_chi2(xs, ys, xp, yp, index_arc, xc, yc, fit_radius, x_shape, y_shape)
                                timer_part1_areadiff += time.time()-timer
                                
                                #If it's better than the last one we randomly
                                # found, store its chi-squared value and store
                                # the set of polygon vertices.
                                if chi < best_chi:
                                    best_chi = chi
                                    xp0, yp0 = xp, yp
                                    x_shape0, y_shape0 = x_shape, y_shape
                                    index_arc_saved = index_arc
                                    #Debugging:
#                                    for point in range(0,len(x_shape)):
#                                        print(str(x_shape[point]) + "\t" + str(y_shape[point]))
#                                    print(str(x_shape[0]) + "\t" + str(y_shape[0]) + "\n")

                                #Get out of this iteration because it is a
                                # convex polygon.
                                break
                            
                            #Failed the convex polygon test, so we need to re-do
                            # it.
                            else:
                                timer_part1_failedConvex += 1

                    timer_part1_total = time.time() - timer_part1_total
                    
                    #Draw the plot for the initial best shape.
                    plotMe(x_shape0,y_shape0,xs,ys,xp0,yp0,n_sides,1.0)

                    #Perform the random walk (this is an area where a real MCMC
                    # will do a better job).  This random walk takes the best
                    # polygon we found from above and massages it, randomly,
                    # less and less as we iterate ("simmer"), to try to improve
                    # it as a plausible fit.
                    timer_part2_total = time.time()
                    simmer = 0.18             #a "simmer" variable to decrease the changes as we iterate
                    n_total_tries = 15000     #how many times will we iterate the fit?
                    flag_improvedThisTime = 0 #used for implementing a poor form of the Metropolis-Hastings MCMC rather than a purely random walk MC
                    print("\nIterating %g times; output will only be updated if there is an improvement." % (n_total_tries))
                    print("# Sides\t\titeration\tchi-sq")
                    print(" ",n_sides, "\t\t", 0, "\t\t", best_chi)
                    fOut = open(inputFile.replace('.csv','')+'-'+str(n_sides)+'-'+str(n_arcs)+'-chi-sq.txt','a')
                    fOut.write("\t\n0\t" + str(best_chi) + "\n")
                    fOut.close()
                    for iCounter_tries in range(0, n_total_tries):
                        #Escape clause.
                        iCounter_escape = 0
                        
                        #We want to make sure we are only picking from convex
                        # polygons.
                        while True:
                            
                            #Every few% more of our total tries, reduce the
                            # random jitter under the assumption that we're
                            # closer and closer to a good solution.  I'm using
                            # such fine gradations because the arc fits are
                            # extremely finicky and can require a large dynamic
                            # range of jittering.
                            timer = time.time()
                            if (iCounter_tries % (int(n_total_tries)/30)) == 0 and (iCounter_tries > 0):
                                simmer *= 0.85

                            #Pick our next points to try.
                            if flag_improvedThisTime == 0:
                                #Perform the random walk on the vertices by
                                # adding random, Gaussian noise with a mean of
                                # the previous best point and a standard
                                # deviation the size of the simmer.
                                xp, yp = np.random.normal(xp0, simmer), np.random.normal(yp0, simmer)
                            
                            else:
                                #Perform a poor form of the Metropolis-Hastings
                                # algorithm, which uses prior improvements to
                                # inform the next random walk step.
                                #TESTING: This gives more CONSISTENT convergence
                                # results but has minimal real convergence speed
                                # as a function of iterations.  As in, there's a
                                # wide envelope from not doing this, but these
                                # are all near the mean. (Testing involved doing
                                # 10 runs on the cone shape and Fejokoo with the
                                # real shapes only.)
                                improvements_x = []
                                improvements_y = []
                                for point in range(0,len(xp0)):
                                    improvements_x.append(improvements[point][0]/2.)
                                    improvements_y.append(improvements[point][1]/2.)
                                xp, yp = np.random.normal(xp0+improvements_x, simmer), np.random.normal(yp0+improvements_y, simmer)
                                
                            #Store for convex shape testing.
                            polygon = np.array([xp, yp]).T
                            timer_part2_makeshape += time.time()-timer

                            #Seed arc information.
                            if n_arcs > 0:
                                #We need to now fit a circle so's to RE-DO the
                                # vertices that will be arc end points.  We have
                                # to do a full-fledged circle fit using all the
                                # points.
                                timer = time.time()
                                center_estimate = np.mean(xp), np.mean(yp)
                                center_2, ier   = optimize.leastsq(f_2, center_estimate)
                                xc, yc          = center_2
                                xc              += np.random.normal(0, simmer/10.)
                                yc              += np.random.normal(0, simmer/10.)
                                Ri_2            = calc_R(*center_2)
                                fit_radius      = Ri_2.mean()+np.random.normal(0, simmer/10.)
                                timer_part2_fitcircle += time.time()-timer
                                
                                #Now re-do the end points of the arc by
                                # calculating the angle they make with the
                                # fitted center and shifting them in radius
                                # based on the fitted radius.
                                timer = time.time()
                                for iCounter_pointsToChange in range(0,n_arcs):
                                    index_vertexInOriginalShape = index_arc_saved[iCounter_pointsToChange] #because I got lost
                                    
                                    #Do the first.
                                    theta = math.atan( (yp[index_vertexInOriginalShape]-yc)/(xp[index_vertexInOriginalShape]-xc) ) *180.0/math.pi
                                    if xp[index_vertexInOriginalShape] < xc: theta += 180.0
                                    xp[index_vertexInOriginalShape] = fit_radius*math.cos(theta*math.pi/180.0)+xc
                                    yp[index_vertexInOriginalShape] = fit_radius*math.sin(theta*math.pi/180.0)+yc
                                    
                                    #Do the second.
                                    if index_arc_saved[iCounter_pointsToChange] == len(xp)-1:
                                        index_vertexInOriginalShape = 0 #because I got lost
                                    else:
                                        index_vertexInOriginalShape = index_arc_saved[iCounter_pointsToChange]+1 #because I got lost
                                    theta = math.atan( (yp[index_vertexInOriginalShape]-yc)/(xp[index_vertexInOriginalShape]-xc) ) *180.0/math.pi
                                    if xp[index_vertexInOriginalShape] < xc: theta += 180.0
                                    xp[index_vertexInOriginalShape] = fit_radius*math.cos(theta*math.pi/180.0)+xc
                                    yp[index_vertexInOriginalShape] = fit_radius*math.sin(theta*math.pi/180.0)+yc
                                timer_part2_moveverticesforcircle += time.time()-timer
                                
                                #And, draw the shape so that we don't have to do
                                # it multiple times, and make sure we get it
                                # right, once.
                                arr_fit = []
                                timer = time.time()
                                if index_arc_saved[0] == -1:
                                    arr_fit = [[xp[i],yp[i]] for i in range(0,len(xp))]
                                else:
                                    for i in range(0,len(xp)):
                                        if (i in index_arc_saved):
                                            arr_fit.append([xp[i],yp[i]])
                                            theta_min     = math.ceil (math.atan( (yp[i  ]-yc)/(xp[i  ]-xc) ) *180.0/math.pi) #ceil so we don't have self-intersections in the geometry by over-shooting
                                            if xp[i] < xc: theta_min += 180.0
                                            if i == len(xp)-1:
                                                theta_max = math.floor(math.atan( (yp[0  ]-yc)/(xp[0  ]-xc) ) *180.0/math.pi) #floor so we don't have self-intersections in the geometry by over-shooting
                                                if xp[0] < xc: theta_max += 180
                                            else:
                                                theta_max = math.floor(math.atan( (yp[i+1]-yc)/(xp[i+1]-xc) ) *180.0/math.pi) #floor so we don't have self-intersections in the geometry by over-shooting
                                                if xp[i+1] < xc: theta_max += 180
                                            if theta_max > theta_min: theta_max -= 360  #we need to go CLOCKWISE, and all these codes go counter-clockwise
                                            theta_max = int(theta_max-1)
                                            theta_min = int(theta_min)
                                            if theta_min < theta_max:
                                                for theta in range(theta_min+1,theta_max-1,angle_fidelity):
                                                    arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                            else:
                                                for theta in range(theta_min-1,theta_max+1,-angle_fidelity):
                                                    arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                        else:
                                            arr_fit.append([xp[i],yp[i]])
                                x_shape = [arr_fit[x][0] for x in range(0,len(arr_fit))]
                                y_shape = [arr_fit[x][1] for x in range(0,len(arr_fit))]
                                timer_part2_makevectorshape += time.time()-timer
                                
                                #Re-store the polygon used to check for
                                # convex-ness.  We don't need to check the
                                # entire shape with arcs because arcs will,
                                # by definition, be convex, and it will slow
                                # the convex test check way down.
                                polygon = np.array([xp, yp]).T
                            
                            #For consistency, if we didn't draw an arc, just
                            # copy the arrays to the _shape ones.
                            else:
                                x_shape = xp.copy()
                                y_shape = yp.copy()
                                
                            #Ensure the test shape is convex -- a requirement of
                            # this problem.
                            timer = time.time()
                            if is_convex_polygon(polygon):
                                timer_part2_checkconvex += time.time()-timer
                                
                                #Calculate the "chi-squared" value for this
                                # polygon.
                                timer = time.time()
                                chi = poly_chi2(xs, ys, xp, yp, index_arc_saved, xc, yc, fit_radius, x_shape, y_shape) #4.0 * n - 2.0 * chi0
                                timer_part2_areadiff += time.time()-timer
                                
                                #If it's better than the previous version, then
                                # store its chi-squared value and store the set
                                # of polygon vertices.  Also output them and
                                # display the result.
                                flag_improvedThisTime = 0
                                if chi < best_chi:
                                    #Store.
                                    flag_improvedThisTime = 1
                                    improvements = [[xp0[point]-xp[point],yp0[point]-yp[point]] for point in range(0,len(yp))]
                                    if not ((len(xp) == len(xp0)) and (len(xp) == (len(yp))) and (len(xp) == len(yp0))):    #getting some weird error occasionally in indexing "improvements" below
                                        flag_improvedThisTime = 0
                                    best_chi = chi
                                    xp0, yp0 = xp, yp   #store the starting point for next time
                                    
                                    #Output.
                                    fOut = open(inputFile.replace('.csv','')+'-'+str(n_sides)+'-'+str(n_arcs)+'-chi-sq.txt','a')
                                    fOut.write(str(iCounter_tries) + "\t" + str(chi) + "\n")
                                    fOut.close()
                                    print(" ",n_sides, "\t\t", iCounter_tries, "\t\t", chi)

                                    #Plot.
                                    plotMe(x_shape,y_shape,xs,ys,xp0,yp0,n_sides,1e-10)
                                
                                #Get out of this iteration because it is a
                                # convex polygon.
                                break
                            
                            #Failed the convex polygon check so need to re-do
                            # this iteration.
                            else:
                                timer_part2_failedConvex += 1
                            
                            #Sometimes, we can just spin our wheels and go on
                            # forever trying to get a non-convex polygon.
                            iCounter_escape += 1
                            if iCounter_escape == 10:
                                flag_improvedThisTime = 0
                                iCounter_escape = 0
                                break
                                
                    #Output timing diagnostics.
                    timer_part2_total = time.time() - timer_part2_total
                    if f_outputTimes == 1:
                        print("\nTime for initial shape search:\t\t\t\t" + str(timer_part1_total))
                        print("Time to pick vertices:\t\t\t\t\t" + str(timer_part1_pickvertices))
                        print("Time to check the angles for minimum spacing:\t\t" + str(timer_part1_checkangles))
                        print("Time to make the shape into proper Python arrays:\t" + str(timer_part1_makeshape))
                        print("Time to pick arcs:\t\t\t\t\t" + str(timer_part1_pickarcs))
                        print("Time to fit a circle:\t\t\t\t\t" + str(timer_part1_fitcircle))
                        print("Time to move the vertices on arcs:\t\t\t" + str(timer_part1_moveverticesforcircle))
                        print("Time to make the shape into a vector:\t\t\t" + str(timer_part1_makevectorshape))
                        print("Time to check for convex shape:\t\t\t\t" + str(timer_part1_checkconvex))
                        print("Time to calculate shape area difference:\t\t" + str(timer_part1_areadiff))
                        print("Had to redo vertex points because spacing was too small:\t" + str(timing_redoForSmallAngles))
                        print("Failed convex test:\t\t\t\t\t\t" + str(timer_part1_failedConvex))
                        print("Time for shape improvement:\t\t\t\t" + str(timer_part2_total))
                        print("Time to make the shape into proper Python arrays:\t" + str(timer_part2_makeshape))
                        print("Time to fit a circle:\t\t\t\t\t" + str(timer_part2_fitcircle))
                        print("Time to move the vertices on arcs:\t\t\t" + str(timer_part2_moveverticesforcircle))
                        print("Time to make the shape into a vector:\t\t\t" + str(timer_part2_makevectorshape))
                        print("Time to check for convex shape:\t\t\t\t" + str(timer_part2_checkconvex))
                        print("Time to calculate shape area difference:\t\t" + str(timer_part2_areadiff))
                        print("Failed convex test:\t\t\t\t\t\t" + str(timer_part2_failedConvex))

                    #Store the information from this iteration.
                    returnArray_Temp = writeReturnArray(n_sides,n_sides_forRealz,n_arcs,best_chi,xc,yc,fit_radius,xp0,yp0,index_arc_saved)
                    returnArray.append(returnArray_Temp)
                    
                    #Store the shapes so we can plot them if we want.
                    array_allshapes_thisone = []
                    for iCounter_vertex in range(0,len(x_shape)):
                        array_allshapes_thisone.append([x_shape[iCounter_vertex],y_shape[iCounter_vertex]])
                    array_allshapes.append(array_allshapes_thisone)
    
    #Return all our fit data.
    return(returnArray, array_allshapes)




#The main enchelada: Take in the rim trace, project it, scale it, fit it, back-
# project the fit, output results.
if __name__ == "__main__":

    #Parse arguments passed to the program.
    parser = argparse.ArgumentParser(description='Fit a polygon to a crater rim.')
    parser.add_argument('--data',     dest='inputFile', action='store', default='',     help='Input .csv file, 2-column, with latitude column labeled "lat" and longitude column labeled "lon".')
    parser.add_argument('--minPoly',  dest='minPoly',   action='store', default='3',    help='Minimum order polygon to try to fit (default 3 = triangle).  Set to 0 to also try to fit a circle')
    parser.add_argument('--maxPoly',  dest='maxPoly',   action='store', default='8',    help='Maximum order polygon to try to fit (default 8 = octagon).')
    parser.add_argument('--minArcs',  dest='minArcs',   action='store', default='0',    help='Minimum number of arcs to try as edges.')
    parser.add_argument('--maxArcs',  dest='maxArcs',   action='store', default='-9',   help='Maximum number of arcs to try as edges. Argument is ignored if >floor(maxPoly/2), and any negative value is a key to let the code go up to that number.')
    parser.add_argument('--repeat',   dest='n_ToDo',    action='store', default='1',    help='How many times to do the fit attempt, to test for how strong the fit really is.  1 = do ONCE, not repeat once, so 2 will do the fit twice, etc.  Exception: Circles are only fit once.')
    parser.add_argument('--ellipsoid',dest='ellipsoid', action='store', default='WGS84',help='Ellipsoid to use.  Options are: Earth\'s "WGS84", Mercury, Venus, Moon, Mars, Vesta, Ceres.')
    args=parser.parse_args()


    #Read in the crater data to a Pandas dataframe.
    d = pd.read_csv(args.inputFile)
    
    
    #We need to project the crater data into a local projection, in kilometers.
    # To do that, we're going to calculate Great Circle bearings between the
    # center and each point, then calculate Great Circle distances.  To get the
    # new coordinates, we're going to then use Cartesian trigonometry, treating
    # the angle as the GC bearing and the hypotenuse as GC distance.
 
    #Part 1: Extract the crater latitude/longitude into separate arrays.
    lon_degrees, lat_degrees = np.array(d['lon']), np.array(d['lat'])
    center_lon_degrees = np.mean(lon_degrees)
    center_lat_degrees = np.mean(lat_degrees)
    
    #Part 2: Calculate Great Circle bearings.
    bearings = [sphere.bearing((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point])) for iCounter_point in range(0,len(lon_degrees))]
    
    #Part 3: Calculate Great Circle distances.
    #Not sure how to get the passage of a structure, so using lots of if-else:
    if args.ellipsoid == "Mercury":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Mercury) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Mercury.a/360.
    elif args.ellipsoid == "Venus":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Venus) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Venus.a/360.
    elif args.ellipsoid == "WGS84" or args.ellipsoid == "Earth":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=WGS84) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*WGS84.a/360.
    elif args.ellipsoid == "Moon":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Moon) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Moon.a/360.
    elif args.ellipsoid == "Mars":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Mars) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Mars.a/360.
    elif args.ellipsoid == "Vesta":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Vesta) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Vesta.a/360.
    elif args.ellipsoid == "Ceres":
        distances = [ellipsoid.py_distance((center_lon_degrees,center_lat_degrees),(lon_degrees[iCounter_point],lat_degrees[iCounter_point]),ellipsoid=Ceres) for iCounter_point in range(0,len(lon_degrees))]
        projection  = 2*math.pi*Ceres.a/360.

    #Aside:  Another issue with this code is that all points MUST go clock-wise,
    # and there can't be any go-backs nor overlaps.  So, we're going to sort
    # these arrays by bearing.
    bearings, distances = (list(t) for t in zip(*sorted(zip(bearings, distances))))
    
    #Part 4: Calculate new locations.
    lon_km = [distances[iCounter_point] * math.sin(bearings[iCounter_point] * math.pi/180.) for iCounter_point in range(0,len(lon_degrees))]
    lat_km = [distances[iCounter_point] * math.cos(bearings[iCounter_point] * math.pi/180.) for iCounter_point in range(0,len(lon_degrees))]
    
    #Normalize the rim to a roughly unit square, but centered at the origin.
    unit_scale = np.mean([(np.amax(lat_km)-np.amin(lat_km)),(np.amax(lon_km)-np.amin(lon_km))])
    lon_scaled = lon_km / unit_scale
    lat_scaled = lat_km / unit_scale
    
    
    #####----------#####

    #We're going to use a determinist method to try to identify some possible
    # hinges/joints to use as seeds for the polygon-finder code.  This is long,
    # but it can be very useful.
    #Incidentally, this was the original code before the Monte Carlo approach.
    
    #With everything projected now into Cartesian space, calculate a few things,
    # repurposing some of the above vectors because we don't need them anymore:
    # - Smoothed rim trace.
    # - Distances between rim points.
    # - Bearings from one rim point ot the next.
    # - Integrated distance along the rim trace.
    # - Derivative of the bearings from one point to the next.
    distances = [((lat_km[iCounter_point]-lat_km[(iCounter_point+1)%(len(lon_km)-1)])**2+(lon_km[iCounter_point]-lon_km[(iCounter_point+1)%(len(lon_km)-1)])**2)**0.5 for iCounter_point in range(0,len(lon_km))]
    smoothing_distance = int(len(distances)/15) if int(len(distances)/15)%2 == 1 else int(len(distances)/15)+1 #must be an odd number
    smoothing_distance = 5 if smoothing_distance < 5 else smoothing_distance    #must be >3 since that's our polygon order in the filter
    lon_km_smoothed = savgol_filter(lon_km,smoothing_distance,3, mode='wrap')
    lat_km_smoothed = savgol_filter(lat_km,smoothing_distance,3, mode='wrap')
    distances = [((lat_km_smoothed[iCounter_point]-lat_km_smoothed[(iCounter_point+1)%(len(lon_km_smoothed)-1)])**2+(lon_km_smoothed[iCounter_point]-lon_km_smoothed[(iCounter_point+1)%(len(lon_km_smoothed)-1)])**2)**0.5 for iCounter_point in range(0,len(lon_km_smoothed))]
    bearings  = [math.atan( (lat_km_smoothed[iCounter_point]-lat_km_smoothed[(iCounter_point+1)%(len(lon_km_smoothed)-1)]) / (lon_km_smoothed[iCounter_point]-lon_km_smoothed[(iCounter_point+1)%(len(lon_km_smoothed)-1)]+random.uniform(-1e-10,1e-10)) ) *180./math.pi for iCounter_point in range(0,len(lon_km_smoothed))]
    bearings  = [(90.-bearings[iCounter_point] if (lon_km_smoothed[iCounter_point] < lon_km_smoothed[(iCounter_point+1)%(len(lon_km_smoothed)-1)]) else 360. - (90.+bearings[iCounter_point])) for iCounter_point in range(0,len(lon_km_smoothed))]
    distances_INT = distances.copy()
    for iCounter_point in range(1,len(lon_km_smoothed)+1):
        distances_INT[(iCounter_point+0)%(len(lon_km_smoothed)-1)] += distances_INT[(iCounter_point-1)%(len(lon_km_smoothed)-1)]    #integrate
        if iCounter_point < len(lon_km_smoothed):
            bearings[(iCounter_point+0)%(len(lon_km_smoothed)-1)]   = bearings[iCounter_point]+360. if np.abs(bearings[iCounter_point]-bearings[iCounter_point-1])>180. else bearings[iCounter_point] #make the bearing continuous when we go over +180°
    bearings_DIF  = bearings.copy()
    bearings_DIF[0] = bearings[1]-bearings[0]
    bearings_DIF[len(bearings_DIF)-1] = bearings[len(bearings_DIF)-1]-bearings[len(bearings_DIF)-2]
    for iCounter_point in range(1,len(bearings_DIF)-2): #use central difference method to differentiate bearings
        bearings_DIF[iCounter_point] = ( (bearings[iCounter_point+1]-bearings[iCounter_point]) + (bearings[iCounter_point]-bearings[iCounter_point-1]) ) / 2.

    #Vectors and Variables.
    array_sides  = []       #stores as a tuple the indices of the start and end of any edge
    array_angles = []       #stores the average bearing of any found edge
    array_length = []       #stores the length along the rim for any found edge
    counter_point_start = 0
    tolerance_distance_min_forside  = 10    #km
    tolerance_angle_max_forside     = 10    #degrees
    tolerance_distance_max_forhinge = 5     #km
    tolerance_angle_min_forhinge    = 20    #degrees

    #Large loop to do the math.  This loop will walk around the crater rim and,
    # based on the four tolerances, will determine if and where there are any
    # polygonal edges and/or hinges.  Rather than explain how it works up here,
    # I will walk you through it as we go.  To start off with, we are going to
    # loop through each point along the rim and determine if there are any edges
    # or hinges from each point.  But, if we find one, we skip to the end of it
    # to determine any remaining, such that this is set up as a while-True loop
    # instead of for() loop because Python does not allow you to dynamically
    # alter the iterating variable within the loop itself.
    while True:

        #For the initial search from this starting point, we need the index of
        # the first possible end point for this edge which is based on distance.
        counter_point_end = round(np.searchsorted(distances_INT, distances_INT[counter_point_start]+float(tolerance_distance_min_forside)))

        #NumPY will NOT return an error if the search is before or after the
        # list, so we need to check for that.
        if (counter_point_end > 0) and (counter_point_end < len(distances_INT)):

            #We have a set of points that could be an edge because it's long
            # enough. The first step in testing it is to calculate the standard
            # deviation. In this calculation, we want the end points to be
            # inclusive, so we need to slice up to +1.  We also want the sample
            # standard deviation, not the population standard deviation, so need
            # to use ddof=1.
            standardDeviation = np.std(bearings[counter_point_start:counter_point_end+1], ddof=1)

            #Now, test that standard deviation.
            if standardDeviation <= float(tolerance_angle_max_forside):

                #We successfully found points that can be considered a side, so
                # now want to look further along the rim to determine if any
                # more contiguous points could be considered part of this side,
                # too.
                while True:
                    counter_point_end += 1
                    standardDeviation = np.std(bearings[counter_point_start:counter_point_end+1], ddof=1)
                    if (standardDeviation > float(tolerance_angle_max_forside)) or (counter_point_end >= len(lon_km_smoothed)-2):
                        counter_point_end -= 1  #subtract 1 because we went over
                        break

                #We have our maximum-length rim section that qualifies as an
                # edge, so now re-calculate the standard deviation of the
                # bearings of the points within it.
                reference_standardDeviation = np.std(bearings[counter_point_start:counter_point_end+1], ddof=1)

                #See if shifting the edge back-and-forth at all allows it to be
                # extended or shifted to better represent an edge.
                shift_start = +1
                shift_end   = +1 if counter_point_end < len(lon_km_smoothed)-2 else 0
                reference_length = distances_INT[counter_point_end+1]-distances_INT[counter_point_start]
                while True:

                    #Start off by shifting 1 point along the rim.
                    counter_point_start_test = counter_point_start + shift_start
                    counter_point_end_test   = counter_point_end   + shift_end

                    #Ensure the length is still long enough for our threshold;
                    # of not, increase the end point until it is.
                    if distances_INT[counter_point_end_test+1]-distances_INT[counter_point_start_test] < float(tolerance_distance_min_forside):
                        while True:
                            shift_end += 1
                            counter_point_end_test = counter_point_end + shift_end
                            if counter_point_end_test >= len(lon_km_smoothed)-1:
                                shift_end -= 1
                                break   #break if we've gone over the end of the rim
                            if distances_INT[counter_point_end_test+1]-distances_INT[counter_point_start_test] >= float(tolerance_distance_min_forside):
                                break   #break if it's long enough
                            if counter_point_end + shift_end >= len(lon_km_smoothed):
                                break   #break if we go over the edge -- TO DO: Make wrap-around aware.

                    #Now check to see if the standard deviation of the bearings
                    # both meets our requirements for a maximum to still be a
                    # side, and is better than the original side we found.  If
                    # it is, then set new reference values and set the shifts to
                    # go further.  If not then decrease the shift back to the
                    # previous loop, and break.
                    standardDeviation = np.std(bearings[counter_point_start_test:counter_point_end_test+1], ddof=1)
                    if (standardDeviation <= float(tolerance_angle_max_forside)) and (standardDeviation < reference_standardDeviation):
                        reference_length = distances_INT[counter_point_end_test+1]-distances_INT[counter_point_start_test]
                        reference_standardDeviation = np.std(bearings[counter_point_start_test:counter_point_end_test+1], ddof=1)
                        shift_start += 1
                        if counter_point_end_test + shift_end < len(distances_INT)-2: shift_end += 1
                    else:
                        shift_start -= 1
                        shift_end   -= 1
                        break
                
                #Set the start/end points to the results from above.
                counter_point_start += shift_start
                counter_point_end   += shift_end

                #Determine if we can correct back at all, possibly extending
                # either start or end by one point.  We still need to check ang-
                # ular variation, but since we are EXTENDING the sides, we don't
                # need to check for length.
                flag_start_decrease = 0
                flag_stop_increase  = 0
                if (counter_point_start >= 1) and (counter_point_start > (array_sides[len(array_sides)-1][1] if len(array_sides)>0 else 0)):    #TO DO: Make wrap-around-aware
                    standardDeviation = np.std(bearings[counter_point_start-1:counter_point_end+1], ddof=1)
                    if standardDeviation <= float(tolerance_angle_max_forside):
                        flag_start_decrease -= 1
                    counter_point_start += flag_start_decrease
                if counter_point_end+2 <= len(lon_km_smoothed):    #TO DO: Make wrap-around-aware
                    standardDeviation = np.std(bearings[counter_point_start:counter_point_end+2], ddof=1)
                    if standardDeviation <= float(tolerance_angle_max_forside):
                        flag_stop_increase  += 1
                    counter_point_end   += flag_stop_increase
                reference_standardDeviation = np.std(bearings[counter_point_start:counter_point_end+1], ddof=1)
                reference_length = distances_INT[counter_point_end+1]-distances_INT[counter_point_start]

                #Now that we have the for-realz edge start/end indices, store
                # them.
                array_sides.append([counter_point_start,counter_point_end+1])

                #Since we have a real edge, store the average bearings of this
                # side.
                array_angles.append(np.mean(bearings[counter_point_start:counter_point_end+1]))

                #Since we have a real edge, store the length along the rim for
                # it.
                array_length.append(distances_INT[counter_point_end+1]-distances_INT[counter_point_start])

                #Set up to testfor another edge at the end of this one.
                counter_point_start = counter_point_end+1

            #The standard deviation of the minimum-length side was too large, so
            # it does not count as an edge and we have to move on, starting with
            # the next point as the possible start location.
            else:
                counter_point_start += 1

        #The distance measure for an edge failed to find something within the
        # list, so we should move on.
        #TBD: This is where wrap-around code needs to be developed.
        else:
            counter_point_start += 1

        #Our only, singular quit criterion.
        if counter_point_start > len(distances)-1:
            break

    #Now determine if the possible hinges meet the criteria set by the arguments
    # for maximum distance and minimum angle.  Special case for the last
    # candidate hinge to support wrap-around.
    array_hinge_valid = [0]*(len(array_angles)) #holds a boolean array
    if len(array_angles) > 0:

        #All angles but the last edge to first edge.
        for counter_hinge in range(0,len(array_angles)-1):
            if(distances_INT[array_sides[counter_hinge+1][0]]-distances_INT[array_sides[counter_hinge][1]] < float(tolerance_distance_max_forhinge)):
                if(array_angles[counter_hinge+1]-array_angles[counter_hinge] > float(tolerance_angle_min_forhinge)):
                    array_hinge_valid[counter_hinge] = 1

        #Wrap-around for the last angle.
        if(distances_INT[array_sides[0][0]]+(distances_INT[len(distances_INT)-1]-distances_INT[array_sides[len(array_angles)-1][1]]) < float(tolerance_distance_max_forhinge)):
            if((array_angles[0]+360)-array_angles[len(array_angles)-1] > float(tolerance_angle_min_forhinge)):
                array_hinge_valid[len(array_hinge_valid)-1] = 1

    #Shrink these vertex points down to possible indices to try.
    array_vertex_points_ToSeed = []
    for iCounter_vertex in range(0,len(array_sides)):
        if iCounter_vertex > 0:
            if array_sides[iCounter_vertex][0] not in array_vertex_points_ToSeed:
                array_vertex_points_ToSeed.append(array_sides[iCounter_vertex][0])
        if iCounter_vertex < len(array_sides)-1:
            if array_sides[iCounter_vertex][1] not in array_vertex_points_ToSeed:
                array_vertex_points_ToSeed.append(array_sides[iCounter_vertex][1])
    
    #####----------#####


    #Do the fitting, in our projected, normalized reference frame.
    returnedArray, returnedShapes = fit_poly(lon_scaled, lat_scaled, int(args.minPoly), int(args.maxPoly), int(args.minArcs), int(args.maxArcs), int(args.n_ToDo), args.inputFile, np.array(array_vertex_points_ToSeed))


    #Back-normalize and output the results.
    projection /= unit_scale
    for iCounter_shapes in range(0,len(returnedArray)):
        returnedArray_ThisShape = returnedArray[iCounter_shapes]
        f_arcs = 0
        circle_info = returnedArray_ThisShape[len(returnedArray_ThisShape)-2]
        
        #Calculate any circle data.
        if len(circle_info) > 0:
            f_arcs = 1
            circle_lat = returnedArray_ThisShape[len(returnedArray_ThisShape)-2][1]/projection+center_lat_degrees
            circle_lon = returnedArray_ThisShape[len(returnedArray_ThisShape)-2][0]/(projection*math.cos(circle_lat*math.pi/180))+center_lon_degrees
            circle_rad = returnedArray_ThisShape[len(returnedArray_ThisShape)-2][2]*unit_scale/1e3
            
        #Output data about each vertex and side.
        if len(returnedArray_ThisShape) > 2:
            if len(returnedArray_ThisShape) > len(lon_degrees):
                fOut = open(args.inputFile.replace('.csv','')+'-edges-circle.txt','a') #if it's >max-n-gon, it's a circle
            else:
                fOut = open(args.inputFile.replace('.csv','')+'-edges.txt','a')
            for iCounter_points in range(len(returnedArray_ThisShape)-2):   #iterate through each vertex point
                #We now need to back-project.  To use Vincenty's "Direct"
                # method to do this, we need a starting location in degrees,
                # the bearing in degrees, and the distance to travel in a
                # physical unit (meters, since everything else is in meters at
                # this point).  The starting point is simply the center we found
                # before during the scaling process.
                
                #To calculate distance, we need to get our vertex points, back-
                # project the scaled units, but keep them in kilometers.  We can
                # then assume that these are from the center of the shape at
                # (0,0), so use basic hypotenuse calculations for a right
                # triangle to get the distance to them.
                thisPoint_lon = returnedArray_ThisShape[iCounter_points][0]*unit_scale
                thisPoint_lat = returnedArray_ThisShape[iCounter_points][1]*unit_scale
                distance      = math.sqrt(thisPoint_lat**2 + thisPoint_lon**2)   #assume Cartesian coordinates are accurate
                
                #For bearing, we need to use our vertex points and, assuming
                # things in this Cartesian space properly preserve angles (which
                # they should since we used Great Circle bearings to get here in
                # the first place), we can use simple trigonometry to calculate
                # the angle from (0,0) to the vertex.
                bearing       = math.atan(thisPoint_lat/thisPoint_lon) *180./math.pi #assume Cartesian angles are accurate
                bearing       = 90.-bearing if (thisPoint_lon > 0) else 360. - (90.+bearing) #convert the ±90° to 0–360° after rotating so 0° is North, not East
                
                #To get the actual lat/lon coordinates of the vertices, we will
                # need to use Vincenty's formulae, using the center lat/lon in
                # degrees, the bearing clockwise from north that we just
                # calculated, and the distance of the point, in km, from the
                # center, that we also just calculated.  The .py_endCoordinates
                # function returns an array of lat/lon/bearing where bearing
                # is the final bearing (since, on a sphere, start bearing is
                # slightly different from the end ... FUN!!
                if args.ellipsoid == "Mercury":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Mercury))
                elif args.ellipsoid == "Venus":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Venus))
                elif args.ellipsoid == "WGS84" or args.ellipsoid == "Earth":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=WGS84))
                elif args.ellipsoid == "Moon":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Moon))
                elif args.ellipsoid == "Mars":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Mars))
                elif args.ellipsoid == "Vesta":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Vesta))
                elif args.ellipsoid == "Ceres":
                    newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Ceres))
                
                #Finally, calculate the actual bearing we're interested in: The
                # direction from one vertex point to another.  For this, we're
                # going to go back to the Cartesian coordinates under the
                # assumption that it has preserved angles, at least better than
                # forward- and back-projecting will.  To do this math, we're
                # just going to calculate the difference between the bearing TO
                # this point's vertex and the next one.
                nextPoint_lat = returnedArray_ThisShape[(iCounter_points+1)%(len(returnedArray_ThisShape)-2)][1]*unit_scale
                nextPoint_lon = returnedArray_ThisShape[(iCounter_points+1)%(len(returnedArray_ThisShape)-2)][0]*unit_scale
                bearing_side  = math.atan( (nextPoint_lat-thisPoint_lat) / (nextPoint_lon-thisPoint_lon+random.uniform(-1e-10,1e-10)) ) *180./math.pi #add random noise to avoid division by zero
                bearing_side  = 90.-bearing_side if (nextPoint_lon > thisPoint_lon) else 270.-bearing_side #convert the ±90° to 0–360° after rotating so 0° is North, not East
                
                #Output.
                if len(returnedArray_ThisShape) > len(lon_degrees):
                    fOut.write(str(newData[0])+"\t"+str(newData[1]) + "\n")
                else:
                    fOut.write(str(newData[0])+"\t"+str(newData[1])+"\t"+str(returnedArray_ThisShape[iCounter_points][2])+"\t"+str(bearing_side) + "\n")
            fOut.write("\n")
            fOut.close()
        
        #Output information about the circle if any arcs are present.
        if f_arcs == 1:
            fOut = open(args.inputFile.replace('.csv','')+'-circdata.txt','a')
            fOut.write(str(circle_lon)+"\t"+str(circle_lat)+"\t"+str(circle_rad) + "\n")
            fOut.close()
#            print("\tArc Data (lon/lat/rad):\t"+str(circle_lon)+"\t"+str(circle_lat)+"\t"+str(circle_rad))
#        print("\n")


    #Back-normalize and output the *shapes*.
    projection /= unit_scale
    fOut = open(args.inputFile.replace('.csv','')+'-shape_forDisplay.txt','a')
    for iCounter_shapes in range(0,len(returnedShapes)):
        returnedArray_ThisShape = returnedShapes[iCounter_shapes]
        for iCounter_points in range(len(returnedArray_ThisShape)):   #iterate through each vertex point
            #We now need to back-project.  To use Vincenty's "Direct" method to
            # do this, we need a starting location in degrees, the bearing in
            # degrees, and the distance to travel in a physical unit (meters,
            # since everything else is in meters at this point).  The starting
            # point is simply the center we found before during the scaling
            # process.
            
            #To calculate distance, we need to get our vertex points, back-
            # project the scaled units, but keep them in kilometers.  We can
            # then assume that these are from the center of the shape at (0,0),
            # so use basic hypotenuse calculations for a right triangle to get
            # the distance to them.
            thisPoint_lon = returnedArray_ThisShape[iCounter_points][0]*unit_scale
            thisPoint_lat = returnedArray_ThisShape[iCounter_points][1]*unit_scale
            distance      = math.sqrt(thisPoint_lat**2 + thisPoint_lon**2)   #assume Cartesian coordinates are accurate
            
            #For bearing, we need to use our vertex points and, assuming things
            # in this Cartesian space properly preserve angles (which they
            # should since we used Great Circle bearings to get here in the
            # first place), we can use simple trigonometry to calculate the
            # angle from (0,0) to the vertex.
            bearing       = math.atan(thisPoint_lat/thisPoint_lon) *180./math.pi #assume Cartesian angles are accurate
            bearing       = 90.-bearing if (thisPoint_lon > 0) else 360. - (90.+bearing) #convert the ±90° to 0–360° after rotating so 0° is North, not East
            
            #To get the actual lat/lon coordinates of the vertices, we will need
            # to use Vincenty's formulae, using the center lat/lon in degrees,
            # the bearing clockwise from north that we just calculated, and the
            # distance of the point, in km, from the center, that we also just
            # calculated.  The .py_endCoordinates function returns an array of
            # lat/lon/bearing where bearing is the final bearing (since, on a
            # sphere, start bearing is slightly different from the end).
            if args.ellipsoid == "Mercury":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Mercury))
            elif args.ellipsoid == "Venus":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Venus))
            elif args.ellipsoid == "WGS84" or args.ellipsoid == "Earth":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=WGS84))
            elif args.ellipsoid == "Moon":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Moon))
            elif args.ellipsoid == "Mars":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Mars))
            elif args.ellipsoid == "Vesta":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Vesta))
            elif args.ellipsoid == "Ceres":
                newData = (ellipsoid.py_endCoordinates([center_lon_degrees,center_lat_degrees],bearing,distance,ellipsoid=Ceres))
            
            #Finally, calculate the actual bearing we're interested in: The
            # direction from one vertex point to another.  For this, we're going
            # to go back to the Cartesian coordinates under the assumption that
            # it has preserved angles, at least better than forward- and back-
            # projecting will.  To do this math, we're just going to calculate
            # the difference between the bearing TO this point's vertex and the
            # next one.
            nextPoint_lat = returnedArray_ThisShape[(iCounter_points+1)%(len(returnedArray_ThisShape)-1)][1]*unit_scale
            nextPoint_lon = returnedArray_ThisShape[(iCounter_points+1)%(len(returnedArray_ThisShape)-1)][0]*unit_scale
            bearing_side  = math.atan( (nextPoint_lat-thisPoint_lat) / (nextPoint_lon-thisPoint_lon+random.uniform(-1e-10,1e-10)) ) *180./math.pi   #add random noise to avoid division by zero
            bearing_side  = 90.-bearing_side if (nextPoint_lon > thisPoint_lon) else 270.-bearing_side #convert the ±90° to 0–360° after rotating so 0° is North, not East
            
            #Output.
            fOut.write(str(newData[0])+"\t"+str(newData[1]) + "\n")
            if iCounter_points == 0: #repeat the end for display purposes
                saved = str(newData[0])+"\t"+str(newData[1]) + "\n"
        fOut.write(saved)
        fOut.write("\n") #separate from the next shape
    fOut.close()



#    #Also output perpendicular bearings from all original rim point segments.
#    for iCounter_vertex in range(0,len(lon)):
#        thisPoint_lon = lon[iCounter_vertex]
#        thisPoint_lat = lat[iCounter_vertex]
#        nextPoint_lon = lon[(iCounter_vertex+1)%len(lon)]
#        nextPoint_lat = lat[(iCounter_vertex+1)%len(lon)]
#        bearing = sphere.bearing((thisPoint_lon,thisPoint_lat),(nextPoint_lon,nextPoint_lat))
#        print(bearing)
