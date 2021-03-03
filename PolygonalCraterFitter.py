import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize
#from shapely.geometry import Point
from shapely.geometry.polygon import Polygon #conda install shapely
import matplotlib.path as mpltPath
import pandas as pd
import math
import random
#import emcee    #conda install -c astropy emcee



#Attempt to calculate some meaningful chi-squared metric for the polygon shape.
# At the moment, this is an area argument where we take the sum of the area
# between the polygon and any part of the rim interior to it, and between the
# polygon and any part of the rim exterior to it.  As the polgyon fit improves,
# then the difference between the rim and polygon should decrease.
def poly_chi2(x1, y1, x2, y2, index_arc, xc, yc, r, x_shape, y_shape):

#    #Get the absolute extents to minimize area calculated.
#    # TBD: This won't account for arcs that go beyond these min/max.
#	xmin, xmax = min(x1.min(), x2.min()), max(x1.max(), x2.max())
#	ymin, ymax = min(y1.min(), y2.min()), max(y1.max(), y2.max())
 
    #What we want as a metric of how good this shape is to represent the crater
    # is the area of the non-overlap area of each polygon.  Since if they
    # overlap perfectly, it's a perfect fit!  Fortunately, the shapely library
    # can do this for us.
    arr_rim = [[x1[i],y1[i]] for i in range(0,len(x1))]
    shp_rim = Polygon(arr_rim)
    
    #Parse the polygon representation of the fit.
    arr_fit = [[x_shape[i],y_shape[i]] for i in range(0,len(x_shape))]
    shp_fit = Polygon(arr_fit)

    if shp_fit.is_valid:
        difference = shp_rim.symmetric_difference(shp_fit)
        area = difference.area
    else:
        area = 1E100

    return area




'''
def lnlike(params, x2, y2):
	x1 = params[0:6]
	y1 = params[6:]

	p = np.array([x1, y1]).T 

	if not is_convex_polygon(p):
		return -np.inf

	return -0.01 * poly_chi2(x1, y1, x2, y2)

'''


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
    TWO_PI = 2 * np.pi

    try:  # needed for any bad points or direction changes
        # Check for too few points
        #if len(polygon) < 3:
        #    return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = np.arctan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > np.pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon



  
  
#Primary fitting algorithm.
def fit_poly(xs, ys):

    #Placeholder chi-squared value that is large relative to what we expect.
    best_chi = 1E100
    
    #Scale for chi-squared because they can be huge, and e^huge = NaN.
    scale_chi = 1e2

    #Alex's naïve Monte Carlo program that tries to do a best-fit and does a
    # reasonable job, but (he says) could be significantly improved upon.
    
    #Test polygons with different numbers of sides.
    n_side_min = 0
    n_side_max = 0
    for n_sides in range(n_side_min, n_side_max+1):
        for n_arcs in range(0,int(math.floor(n_sides/2))+1):
            #We are testing not only for polygon sides, but also the potential
            # that some sides are better represented by circle arcs.  That means
            # the real number of polygon sides could be less.  We test for up
            # to n_sides-2 arcs, meaning that for 4 "sides" we will test for
            # 4 sides, 3 sides 1 arc, 2 sides 2 arcs.
            n_sides_forRealz = n_sides-n_arcs
            
            print("Trying to fit a %g-sided polygon with %g arc(s) instead of straight side(s)." % (n_sides, n_arcs))

            #Placeholder chi-squared value that is large relative to what we expect
            # but will store the best chi-squared for this polygon test.
            best_chi    = 1E100
            
            #We're going to do something different if n_sides == 0 (it's a circle)
            if n_sides > 0:
                #Perform an initial, random guess at the best-fit polygon.  Store the
                # best one as determined by the chi-squared criterion.
                n_random_guesses_for_initial_shape = 1000*n_sides #how many initial, completely random polygons to try [for the cone, at 500, only 70% [N=20 tests] converged; at 2500, 95% [N=20 tests] started close enough to converge; for Fekojoo, 500 was enough for all of them as hexagons]
                print("Creating an initial guess for polygon vertices by testing %g random versions." % (n_random_guesses_for_initial_shape))
                for iCounter_guesses in range(0, n_random_guesses_for_initial_shape):
                
                    #We want to make sure we are only picking from convex polygons.
                    while True:
                    
                        #We don't want picks to be truely random: We want some
                        # spacing between vertices.
                        while True:
                            #Pick N-sides random indices from the crater rim to be
                            # the random vertices and sort.
                            inds = np.random.choice(np.arange(0, xs.size, 1), n_sides, replace=False)
                            inds.sort()
                            
                            #Figure out the angles.  The first, diffs1, uses angles
                            # from –90° to +270°, where –90°/+270° is due south and
                            # angles go counter-clockwise.  The second, diffs2,
                            # wraps things to a 0° to +360° domain and so will make
                            # sure that we factor in points just slightly on
                            # opposite sides of –90°/+270°.
                            xc, yc = np.mean(xs), np.mean(ys)   #simple center of the RIM TRACE
                            angles = [(math.atan((ys[inds[counter]]-yc)/(xs[inds[counter]]-xc))*180.0/math.pi) for counter in range(0,len(inds))]
                            for iCounter_angle in range(0,len(angles)):
                                if xs[inds[iCounter_angle]] < xc: angles[iCounter_angle] += 180.0
                            diffs1 = [abs(angles[counter]-angles[counter+1]) for counter in range(0,len(angles)-1)]
                            diffs1.append(abs(angles[len(angles)-1]-angles[0]))
                            for iCounter_angle in range(0,len(angles)):
                                if angles[iCounter_angle]   < 0 : angles[iCounter_angle] = 360 + angles[iCounter_angle]
                            diffs2 = [abs(angles[counter]-angles[counter+1]) for counter in range(0,len(angles)-1)]
                            diffs2.append(abs(angles[len(angles)-1]-angles[0]))
                            if (np.amin(diffs1) > (360.0/n_sides)*0.333333) and (np.amin(diffs2) > (360.0/n_sides)*0.333333):
                                break
                        
                        #Jitter the locations of those random indices from the
                        # original rim to account for noise and offsets.
                        xp, yp = np.random.normal(xs[inds], 0.05), np.random.normal(ys[inds], 0.05)
                        polygon = np.array([xp, yp]).T #will use this for convex testing
                        

                        #Definitions for using scipy.optimize.leastsq for fitting a
                        # circle.  I made these private so that it knows about xp
                        # and yp, which is why they are declared after those.
                        #See: https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html
                        def calc_R(xc, yc):
                            """ calculate the distance of each 2D points from the center (xc, yc) """
                            return ((xp-xc)**2 + (yp-yc)**2)**0.5

                        def f_2(c):
                            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
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
                            #Pick which of the sides is going to be an arc.  This is
                            # slightly complicated in that we have further restrictions:
                            # no two arcs can be next to each other.  That means that we
                            # can't just select all the index points for the arc at
                            # random.
                            index_arc = []
                            index_arc.append(random.randint(0,n_sides-1)) #the arc spans from one index to another, so we need to do a -1 here on the range
                            if n_arcs > 1:
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
                            
                            if flag_failed == 0:
                                #We need to now fit a circle so's to RE-DO the vertices
                                # that will be end points to the arc(s).  We have to do
                                # a full-fledged circle fit using all the points because
                                # randomly picking any three (if there are more than 3)
                                # can result in wildly varying results.
                                center_estimate = np.mean(xp), np.mean(yp)
                                center_2, ier   = optimize.leastsq(f_2, center_estimate)
                                xc, yc          = center_2
                                Ri_2            = calc_R(*center_2)
                                fit_radius      = Ri_2.mean()
                                
                                #Now re-do the end points of the arc by calculating the
                                # angle they make with the fitted center and shifting
                                # them in radius based on the fitted radius.
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
                                
                                #And, draw the shape so that we don't have to do it
                                # multiple times, and make sure we get it right, once.
                                arr_fit = []
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
                                                for theta in range(theta_min+1,theta_max-1,1):
                                                    arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                            else:
                                                for theta in range(theta_min-1,theta_max+1,-1):
                                                    arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                        else:
                                            arr_fit.append([xp[i],yp[i]])
                                x_shape = [arr_fit[x][0] for x in range(0,len(arr_fit))]
                                y_shape = [arr_fit[x][1] for x in range(0,len(arr_fit))]

                                #Re-store the polygon used to check for convex-ness.
                                polygon = np.array([x_shape, y_shape]).T
                        else:
                            x_shape = xp.copy()
                            y_shape = yp.copy()
                        
                        #Ensure the test shape is convex -- a requirement of this problem.
                        if is_convex_polygon(polygon) and (flag_failed == 0):
                            #Calculate the chi-squared value of this test polygon.
                            chi = poly_chi2(xs, ys, xp, yp, index_arc, xc, yc, fit_radius, x_shape, y_shape) #4.0 * n - 2.0 * chi0
                            
                            #If it's better than the last one we randomly found,
                            # store its chi-squared value and store the set of
                            # polygon vertices.
                            if chi < best_chi:
                                best_chi = chi
                                xp0, yp0 = xp, yp
                                x_shape0, y_shape0 = x_shape, y_shape
                                index_arc_saved = index_arc
#                                for point in range(0,len(x_shape)):
#                                    print(str(x_shape[point]) + "\t" + str(y_shape[point]))
#                                print(str(x_shape[0]) + "\t" + str(y_shape[0]) + "\n")

                            #Get out of this iteration because it is a convex
                            # polygon.
                            break


            #Special case for n_sides == 0, to just fit a circle.
            else:
                index_arc = [-1]
                xp              = xs.copy()
                yp              = ys.copy()
                xp0             = xs.copy()
                yp0             = ys.copy()
                def calc_R(xc, yc):
                    """ calculate the distance of each 2D points from the center (xc, yc) """
                    return ((xp-xc)**2 + (yp-yc)**2)**0.5
                def f_2(c):
                    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
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
                print(best_chi)


            #------------#
            #For diagnostic purposes, plot the best version at this point.
            x_shape0 = np.append(x_shape0, x_shape0[0])    #close the polygon
            y_shape0 = np.append(y_shape0, y_shape0[0])    #close the polygon
#            print("Initial guess's chi-squared: ", best_chi)
            plt.clf()
            plt.plot(xs, ys, 'r-')
            plt.plot(x_shape0, y_shape0, 'k-')
            if n_sides > 0: plt.plot(xp0, yp0, 'ko')
            plt.axis('scaled')
            plt.ylim(-0.8,+0.8)
            plt.xlim(-0.8,+0.8)
            plt.draw()
            plt.pause(1.0)
            #------------#


            if n_sides > 0:
                #Perform the random walk (this is an area where a real MCMC will do
                # a better job).  This random walk takes the best polygon we found
                # from above and massages it, randomly, less and less as we iterate,
                # to try to improve it as a plausible fit
                simmer = 0.15             #a "simmer" variable to decrease the changes as we iterate
                n_total_tries = 10000     #how many times will we iterate the fit?
                flag_improvedThisTime = 0 #used for implementing a poor form of the Metropolis-Hastings MCMC rather than a purely random walk MC
                print("\nIterating %g times; output will only be updated if there is an improvement." % (n_total_tries))
                print("# Sides\t\titeration\tchi-sq")
                print(" ",n_sides, "\t\t", 0, "\t\t", best_chi)
                fOut = open('chi-sq.txt','a')
                fOut.write("\t\n0\t" + str(best_chi) + "\n")
                fOut.close()
                for iCounter_tries in range(0, n_total_tries):
                    #Escape clause.
                    iCounter_escape = 0
                    
                    #We want to make sure we are only picking from convex polygons.
                    while True:
                        
                        #Every 2% more of our total tries, reduce the random jitter
                        # under the assumption that we're closer and closer to a
                        # good solution.  I'm using such fine gradations because the
                        # arc fits are extremely finicky and can require a large
                        # dynamic range of jittering.  At the time of this comment,
                        # the range is now 0.125 – 0.00000357... .
                        if (iCounter_tries % (int(n_total_tries)/25)) == 0 and (iCounter_tries > 0):
                            simmer *= 0.8

                        #Pick our next points to try.
                        if flag_improvedThisTime == 0:
                            #Perform the random walk on the vertices by simply
                            # adding random, Gaussian noise with a mean of the
                            # previous best point and a standard deviation the size
                            # of the simmer.
                            xp, yp = np.random.normal(xp0, simmer), np.random.normal(yp0, simmer)
                        
                        else:
                            #Perform a poor form of the Metropolis-Hastings
                            # algorithm, which uses prior improvements to inform the
                            # next random walk step.
                            #TESTING: This gives more CONSISTENT convergence results
                            # but has minimal real convergence speed as a function
                            # of iterations.  As in, there's a wide envelope from
                            # not doing this, but these are all near the mean.
                            # (Testing involved doing 10 runs on the cone shape and
                            # Fekojoo with the real shapes only.)
                            improvements_x = []
                            improvements_y = []
                            for point in range(0,len(xp0)):
                                improvements_x.append(improvements[point][0]/2.)
                                improvements_y.append(improvements[point][1]/2.)
                            xp, yp = np.random.normal(xp0+improvements_x, simmer), np.random.normal(yp0+improvements_y, simmer)
                            
                        #Store for convex shape testing.
                        polygon = np.array([xp, yp]).T
                        
                        #Seed arc information.
                        if n_arcs > 0:
                            #We need to now fit a circle so's to RE-DO the vertices
                            # that will be end points to the arc(s).  We have to do
                            # a full-fledged circle fit using all the points because
                            # randomly picking any three (if there are more than 3)
                            # can result in wildly varying results.
                            center_estimate = np.mean(xp), np.mean(yp)
                            center_2, ier   = optimize.leastsq(f_2, center_estimate)
                            xc, yc          = center_2
                            Ri_2            = calc_R(*center_2)
                            fit_radius      = Ri_2.mean()
                            
                            #Now re-do the end points of the arc by calculating the
                            # angle they make with the fitted center and shifting
                            # them in radius based on the fitted radius.
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
                            
                            #And, draw the shape so that we don't have to do it
                            # multiple times, and make sure we get it right, once.
                            arr_fit = []
                            
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
                                            for theta in range(theta_min+1,theta_max-1,1):
                                                arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                        else:
                                            for theta in range(theta_min-1,theta_max+1,-1):
                                                arr_fit.append([fit_radius*math.cos(theta*math.pi/180.0)+xc,fit_radius*math.sin(theta*math.pi/180.0)+yc])
                                    else:
                                        arr_fit.append([xp[i],yp[i]])
                            x_shape = [arr_fit[x][0] for x in range(0,len(arr_fit))]
                            y_shape = [arr_fit[x][1] for x in range(0,len(arr_fit))]

                            #Re-store the polygon used to check for convex-ness.
                            polygon = np.array([x_shape, y_shape]).T
                        
                        #For consistency, if we didn't draw an arc, just copy the
                        # arrays to the _shape ones.
                        else:
                            x_shape = xp.copy()
                            y_shape = yp.copy()
                            
                        #Ensure the test shape is convex -- a requirement of this problem.
                        if is_convex_polygon(polygon):
                            
                            #Calculate the chi-squared value for this polygon.
                            chi = poly_chi2(xs, ys, xp, yp, index_arc_saved, xc, yc, fit_radius, x_shape, y_shape) #4.0 * n - 2.0 * chi0
                            
                            #If it's better than the previous version, then store
                            # its chi-squared value and store the set of polygon
                            # vertices.  Also output them and display the result.
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
                                fOut = open('chi-sq.txt','a')
                                fOut.write(str(iCounter_tries) + "\t" + str(chi) + "\n")
                                fOut.close()
                                print(" ",n_sides, "\t\t", iCounter_tries, "\t\t", chi)

                                #Plot.
                                x_shape0 = np.append(x_shape, x_shape[0])    #close the polygon for display
                                y_shape0 = np.append(y_shape, y_shape[0])    #close the polygon for display
                                plt.clf()
                                plt.plot(xs, ys, 'r-')
                                plt.plot(x_shape0, y_shape0, 'k-')
                                plt.plot(xp0, yp0, 'ko')
                                plt.axis('scaled')
                                plt.ylim(-0.8,+0.8)
                                plt.xlim(-0.8,+0.8)
                                plt.draw()
                                plt.pause(1e-10)
                            
                            #Get out of this iteration because it is a convex
                            # polygon.
                            break
                        
                        #Sometimes, we can just spin our wheels and go on forever
                        # trying to get a non-convex polygon.
                        iCounter_escape += 1
                        if iCounter_escape == 10:
                            flag_improvedThisTime = 0
                            break

            #Output our estimate for the AIC.  The trick is really figuring out what
            # the liklihood function is.  Alex thinks it's some function of the chi-
            # squared value he's calculating, which is just the area between the
            # polygon and the rim.  That said, the only way I can get AIC to behave
            # they way I expect for Fekojoo, where the 6-sided is the best fit, is
            # to do some 1/exp(chi-squared).  If it's not 1÷, then the value in AIC
            # that is subtracted from the degrees-of-freedom term shrinks as the
            # chi-squared shrinks, when it should be growing because it's a better
            # fit.  If it's not exp() or some other function that increases it more
            # than linearly, then even a factor of 4x decrease in the area between
            # 5 sides and 6 sides does not make this term large enough to decrease
            # this part of the AIC more than the DOF increase.
            if n_sides > 0:
                if n_arcs == 0:
                    d_DOF = 2.0*n_sides #degrees of freedom is just the {x,y} position of each vertex
                else:
                    d_DOF = 2.0*n_sides_forRealz + 2.0*n_arcs + 1 #I'm sure this is wrong, but until I hear otherwise ... the +1 is for the fit center ... but it's not actually free, it's dependent on a circle-fit to all the vertexes, and also the vertexes that end the arcs are now correlated because they must be the same radial distance from the circle center ... but WHERE that arc is (which vertex) is free, so that's another DOF ... yeah, this is complicated.
            else:
                d_DOF = 3.0
            print("Akaike Information Criterion: %f\n" % (2.0*d_DOF - 2.0 * math.log(math.exp(best_chi/d_DOF))))
            print(xp0,yp0)
#            for point in range(0,len(xp)):
#                print(str(xp[point]) + "\t" + str(yp[point]))
#            print(str(xp[0]) + "\t" + str(yp[0]))

            #Also here, or based on AIC, output at the end: Store the best vertex
            # points as well as if they're polygonal or linear, and then at the end
            # output them, their type, and the bearings of edges.
        


#Stuart's notes on this section:  I think all of Alex's code should be replaced
# with my scaled version that projects to proper geodetic distance from the
# center.  For the moment, I have commented out his scaling in order to test.
if __name__ == "__main__":
#    #Create a seed hexagon.
#    t = np.array([0, 1, 2, 3, 4, 5, 6])
#    x = [0, 1, 1.5, 1, 0, -0.5, 0]
#    y = [0, 0, 0.5, 1.0, 1.0, 0.5, 0]
#
#    #Interpoloate seed hexagon.
#    xi, yi = interp1d(t, x, kind='linear'), interp1d(t, y, kind='linear')
#
#    #Create a seed array, [0,6), in 0.01 increments.
#    tv = np.arange(0, 6, 0.01)
#
#    #Create the hexagon again, but with a little bit of noise (the sine waves).
#    xv, yv = xi(tv) + np.sin(tv*10.0)*0.01, yi(tv) + np.sin(tv*33.0 + 4.5)*0.002

    #Read in the crater data to a Pandas dataframe.
    d = pd.read_csv('Fekojoo.csv')
#    d = pd.read_csv('Copernicus.csv')
#    d = pd.read_csv('MeteorCrater.csv')
#    d = pd.read_csv('Mamilia.csv')
#    d = pd.read_csv('Cone.csv')
#    d = pd.read_csv('Mahler.csv')
 
    #Extract the crater latitude/longitude into separate NumPy arrays.
    lon, lat = np.array(d['lon']), np.array(d['lat'])

    #TBD: There MUST be an checking for whether the shape goes clockwise or
    # counter-clockwise.  It is assumed that everything is clockwise and the
    # the code WILL NOT WORK if it's counter-clockwise.
    
    #Normalize the latitude/longitude.
    lon     = lon - np.mean(lon)
    lat     = lat - np.mean(lat)
    scale   = np.mean([(np.amax(lat)-np.amin(lat)),(np.amax(lon)-np.amin(lon))])
    xv      = lon / scale
    yv      = lat / scale

    #Do the fitting, but for yet another normalization of the latitude/longitude(???).
    fit_poly(xv, yv)
