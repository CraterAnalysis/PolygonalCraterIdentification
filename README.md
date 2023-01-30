# PolygonalCraterIdentification
Python code that fits a polygon to an impact crater rim trace.

See the primary PolygonalCraterFitter.py file for additional information, which is repeated below in part:

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
