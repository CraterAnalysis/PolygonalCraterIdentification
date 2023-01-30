from math import pi
from collections import namedtuple

# https://en.wikipedia.org/wiki/Earth_radius
EARTH_MEAN_RADIUS                    = 6371008.8
EARTH_MEAN_DIAMETER                  = 2 * EARTH_MEAN_RADIUS
EARTH_EQUATORIAL_RADIUS              = 6378137.0
EARTH_EQUATORIAL_METERS_PER_DEGREE   = pi * EARTH_EQUATORIAL_RADIUS / 180 # 111319.49079327358
I_EARTH_EQUATORIAL_METERS_PER_DEGREE = 1 / EARTH_EQUATORIAL_METERS_PER_DEGREE

#Mercury
MERCURY_MEAN_RADIUS                     = 2439700 #±1000 meters
MERCURY_MEAN_DIAMETER                   = 2 * MERCURY_MEAN_RADIUS
MERCURY_EQUATORIAL_RADIUS               = 2439700
MERCURY_EQUATORIAL_METERS_PER_DEGREE    = pi * MERCURY_EQUATORIAL_RADIUS / 180.
I_MERCURY_EQUATORIAL_METERS_PER_DEGREE  = 1. / MERCURY_EQUATORIAL_METERS_PER_DEGREE

#Venus
VENUS_MEAN_RADIUS                     = 6051800 #±1000 meters
VENUS_MEAN_DIAMETER                   = 2 * VENUS_MEAN_RADIUS
VENUS_EQUATORIAL_RADIUS               = 6051800
VENUS_EQUATORIAL_METERS_PER_DEGREE    = pi * VENUS_EQUATORIAL_RADIUS / 180.
I_VENUS_EQUATORIAL_METERS_PER_DEGREE  = 1. / VENUS_EQUATORIAL_METERS_PER_DEGREE

#Earth's Moon
MOON_MEAN_RADIUS                     = 1737400 #±? meters
MOON_MEAN_DIAMETER                   = 2 * MOON_MEAN_RADIUS
MOON_EQUATORIAL_RADIUS               = 1738100
MOON_EQUATORIAL_METERS_PER_DEGREE    = pi * MOON_EQUATORIAL_RADIUS / 180.
I_MOON_EQUATORIAL_METERS_PER_DEGREE  = 1. / MOON_EQUATORIAL_METERS_PER_DEGREE

#Mars
MARS_MEAN_RADIUS                     = 3389500 #±200 meters
MARS_MEAN_DIAMETER                   = 2 * MARS_MEAN_RADIUS
MARS_EQUATORIAL_RADIUS               = 3396200
MARS_EQUATORIAL_METERS_PER_DEGREE    = pi * MARS_EQUATORIAL_RADIUS / 180.
I_MARS_EQUATORIAL_METERS_PER_DEGREE  = 1. / MARS_EQUATORIAL_METERS_PER_DEGREE

#Vesta
VESTA_MEAN_RADIUS                     = 262700 #±100 meters
VESTA_MEAN_DIAMETER                   = 2 * VESTA_MEAN_RADIUS
VESTA_EQUATORIAL_RADIUS               = 282450
VESTA_EQUATORIAL_METERS_PER_DEGREE    = pi * VESTA_EQUATORIAL_RADIUS / 180.
I_VESTA_EQUATORIAL_METERS_PER_DEGREE  = 1. / VESTA_EQUATORIAL_METERS_PER_DEGREE

#Ceres
CERES_MEAN_RADIUS                     = 469730 #±100 meters
CERES_MEAN_DIAMETER                   = 2 * CERES_MEAN_RADIUS
CERES_EQUATORIAL_RADIUS               = 482150
CERES_EQUATORIAL_METERS_PER_DEGREE    = pi * CERES_EQUATORIAL_RADIUS / 180.
I_CERES_EQUATORIAL_METERS_PER_DEGREE  = 1. / CERES_EQUATORIAL_METERS_PER_DEGREE


HALF_PI = pi / 2.0
QUARTER_PI = pi / 4.0


# https://en.wikipedia.org/wiki/Geodetic_datum
Datum = namedtuple('Datum', ['a', 'b', 'e', 'f', 'w'])

# https://epsg.io/7030-ellipsoid
WGS84 = Datum(
    a = 6378137.0,  #equatorial radius (semi-major axis)
    b = 6356752.3,  #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.0818192,  #eccentricity                    e = (2*f - f**2)**0.5
    f = 0.0033528,  #flattening                      f = 1-b/a
    w = 7292115e-11,#rotation speed in rad/sec       w = ...
)

Mercury = Datum(
    a = 2439700.0, #equatorial radius (semi-major axis)
    b = 2439700.0, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.0000,    #eccentricity                    e = (2*f - f**2)**0.5
    f = 0,         #flattening                      f = 1-b/a
    w = -99999999, #rotation speed in rad/sec       w = ...
)

Venus = Datum(
    a = 6051800.0, #equatorial radius (semi-major axis)
    b = 6051800.0, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.0000,    #eccentricity                    e = (2*f - f**2)**0.5
    f = 0,         #flattening                      f = 1-b/a
    w = -99999999, #rotation speed in rad/sec       w = ...
)

Moon = Datum(
    a = 1738100.0, #equatorial radius (semi-major axis)
    b = 1736000.0, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.0491424, #eccentricity                    e = (2*f - f**2)**0.5
    f = 0.0012082, #flattening                      f = 1-b/a
    w = -99999999, #rotation speed in rad/sec       w = ...
)

Mars = Datum(
    a = 3396200.0, #equatorial radius (semi-major axis)
    b = 3376200.0, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.1083660, #eccentricity                    e = (2*f - f**2)**0.5
    f = 0.0058889, #flattening                      f = 1-b/a
    w = -99999999, #rotation speed in rad/sec       w = ...
)

Vesta = Datum(
    a = 282450.00, #equatorial radius (semi-major axis) [used average of 572.6 and 557.2 km]
    b = 223200.00, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.6128125, #eccentricity                    e = (2*f - f**2)**0.5
    f = 0.2097716, #flattening                      f = 1-b/a [note: official value uses MAJOR axis, not avg]
    w = -99999999, #rotation speed in rad/sec       w = ...
)

Ceres = Datum(
    a = 482150.00, #equatorial radius (semi-major axis) [used average of 964.4 and 964.2 km]
    b = 445900.00, #polar radius (semi-minor axis)  b = a * (1 - f)
    e = 0.3804149, #eccentricity                    e = (2*f - f**2)**0.5
    f = 0.0751841, #flattening                      f = 1-b/a
    w = -99999999, #rotation speed in rad/sec       w = ...
)
