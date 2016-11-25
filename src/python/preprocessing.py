import sys
import json
import os
import numpy as np

from spaceNet import geoTools as gT
from shapely.geometry import Polygon, shape, Point
#from osgeo import gdal
from osgeo import gdal, osr, ogr

# The path to the data directory where TIFF files located among with WKT
path = sys.argv[1]

# read WKT file with buildings footrpints
buildinglist = gT.readwktcsv(path + 'truth.csv')
# the image dictionary
imageDict = {}
for record in buildinglist:
    image_id = record['ImageId']
    imageList = imageDict.get(image_id)
    if imageList == None:
        imageList = []
        imageDict[image_id] = imageList
    imageList.append(record)

# iterate over collected images and create distance arrays
for image_id, bList in imageDict.iteritems():
    print "Processing image: " + image_id
    # load TIFF images
    ds8 = gdal.Open(path+'8band/'+'8band_'+image_id+'.tif')
    ds3 = gdal.Open(path+'3band/'+'3band_'+image_id+'.tif')

    # the distance array
    dist = np.zeros((ds8.RasterXSize, ds8.RasterYSize))

    # iterate over buildings list in the image
    for record in bList:
        polyGeo = record['poly']
        if polyGeo.IsEmpty():
            print "Empty polygon - skipping!"
            continue
        else:
            print "Processing building #{0}".format(record['BuildingId'])

        #json_poly = json.loads(polyGeo.ExportToJson())
        polygon = polyGeo#shape(json_poly)
        for i in range(ds8.RasterXSize):
            for j in range(ds8.RasterYSize):
                #point = Point(i, j)
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(i, j)
                newpd = point.Distance(polygon)#point.distance(polygon.boundary)
                pd = -100000.0
                if False == polygon.Intersects(point):# polygon.contains(point):
                    newpd = -1.0 * newpd
                if newpd > pd:
                    pd = newpd
                dist[i,j] = pd

    # saving two dimensional array with distance map
    directory = path+'distance/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory + image_id + '.distance', dist)
