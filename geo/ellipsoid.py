from math import degrees, radians, sin, cos, asin, tan, atan, atan2, pi, sqrt, exp, log, fabs
from geo.constants import QUARTER_PI, HALF_PI, WGS84, Mercury, Venus, Moon, Mars, Vesta, Ceres


#1e-12 ≈ 0.06 mm on Earth.
CONVERGENCE_THRESHOLD = 1e-12

#Larger amount needed for Vesta since it's highly elliptical.
MAX_ITERATIONS = 100


#Calculate Great Circle distances, on an ellipsoid (biaxial, not triaxial),
# using the mathematics from Vincenty (1975) (https://en.wikipedia.org/wiki/Vincenty's_formulae).
# This is the "Inverse Problem."
def py_distance(point1, point2, ellipsoid=WGS84):
    
    #Extract latitude and longitude from the input data, converting to radians.
    lon1, lat1 = (radians(coord) for coord in point1)
    lon2, lat2 = (radians(coord) for coord in point2)

    #Some maths.
    U1     = atan((1 - ellipsoid.f) * tan(lat1))
    U2     = atan((1 - ellipsoid.f) * tan(lat2))
    L      = lon2 - lon1
    Lambda = L

    #Some more maths because we have to calculate them a lot.
    sinU1 = sin(U1)
    cosU1 = cos(U1)
    sinU2 = sin(U2)
    cosU2 = cos(U2)

    #Vincenty's method is iterative and must be done until it reaches a
    # convergence criterion or until a maximum number of iterations has been
    # reached.
    for _ in range(MAX_ITERATIONS):
        sinLambda = sin(Lambda)
        cosLambda = cos(Lambda)
        sinSigma = sqrt( (cosU2 * sinLambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
            
        # coincident points
        if sinSigma == 0:
            return 0.0

        cosSigma   = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma      = atan2(sinSigma, cosSigma)
        sinAlpha   = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0

        C         = (ellipsoid.f/16.0) * cosSqAlpha * ( 4.0+ellipsoid.f*(4.0-3.0*cosSqAlpha) )
        LambdaOld = Lambda
        Lambda    = L + (1-C) * ellipsoid.f * sinAlpha * ( sigma + C * sinSigma * ( cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM**2) ) )

        #Check for convergeance.
        if abs(Lambda - LambdaOld) < CONVERGENCE_THRESHOLD:
            break
    
    #We did not converge.
    else:
        return None

    #And now for yet some more maths.
    uSq        = cosSqAlpha * (ellipsoid.a ** 2 - ellipsoid.b ** 2) / (ellipsoid.b ** 2)
    A          = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B          = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s          = ellipsoid.b * A * (sigma - deltaSigma)
    return s


#Given an initial point, an initial bearing, and a distance (s), find the end
# point using the mathematics from Vincenty (1975) (https://en.wikipedia.org/wiki/Vincenty's_formulae).
# This is the "Direct Problem."
def py_endCoordinates(point1, bearing, distance, ellipsoid=WGS84):
    
    #Extract latitude and longitude from the input data, converting to radians.
    lon1, lat1 = (radians(coord) for coord in point1)

    #Convert bearing to radians.
    bearing = radians(bearing)

    #Some maths
    U1       = atan((1 - ellipsoid.f) * tan(lat1))
    sigma1   = atan2(tan(U1), cos(bearing))
    sinAlpha = cos(U1) * sin(bearing)
    uSq      = (1.-sinAlpha**2) * ( (ellipsoid.a**2-ellipsoid.b**2) / (ellipsoid.b**2) )
    AA       = 1 + uSq/16384. * (4096. + uSq * (-768. + uSq * (320. - 175.*uSq)))
    BB       = uSq/1024. * (256. + uSq * (-128. + uSq * (74.-47.*uSq)))
    sigma    = distance / (ellipsoid.b*AA)

    #Vincenty's method is iterative and must be done until it reaches a
    # convergence criterion or until a maximum number of iterations has been
    # reached.
    for _ in range(MAX_ITERATIONS):
        twoSigmaM  = 2*sigma1 + sigma
        cos2SigmaM = cos(twoSigmaM)
        deltaSigma = BB*sin(sigma) * (cos2SigmaM + 0.25*BB * (cos(sigma)*(-1.0+2*(cos2SigmaM)**2) - BB/6.*cos2SigmaM*(-3.+4.*(sin(sigma))**2)*(-3.+4.*(cos2SigmaM)**2)))
        
        sigmaOld = sigma
        sigma    = distance / (ellipsoid.b*AA) + deltaSigma

        #Check for convergeance.
        if abs(sigma - sigmaOld) < CONVERGENCE_THRESHOLD:
            break
    
    #We did not converge.
    else:
        return None

    #And now for yet some more maths.
    lat2       = atan2(sin(U1)*cos(sigma)+cos(U1)*sin(sigma)*cos(bearing) , (1-ellipsoid.f)*sqrt(sinAlpha**2+(sin(U1)*sin(sigma)-cos(U1)*cos(sigma)*cos(bearing))**2))
    Lambda     = atan2(sin(sigma)*sin(bearing) , cos(U1)*cos(sigma)-sin(U1)*sin(sigma)*cos(bearing))
    cosSqAlpha = 1.0 - sinAlpha**2
    CC         = ellipsoid.f/16. * cosSqAlpha * (4. + ellipsoid.f * (4.-3.*cosSqAlpha))
    LL         = Lambda - (1.-CC) * ellipsoid.f * sinAlpha * (sigma + CC*sin(sigma) * (cos2SigmaM + CC*cos(sigma) * (-1.+2.*cos2SigmaM)))
    lon2       = LL + lon1
    bearing2   = atan2(sin(bearing) , -sin(U1)*sin(sigma)+cos(U1)*cos(sigma)*cos(bearing))

    #Return the information in degrees.
    return([degrees(lon2),degrees(lat2),bearing2])
