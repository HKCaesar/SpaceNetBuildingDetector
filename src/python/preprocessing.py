from spaceNet import geoTools as gT
from shapely.geometry import Polygon, shape, Point
import numpy as np
import sys

# The path to the data directory where TIFF files located among with WKT
path = sys.argv[2]

# read WKT file with buildings footrpints
buildinglist = readwktcsv(path + 'truth.csv')
for record in buildinglist:
    image_id = record['ImageId']
    print "Processing image: " + image_id
    # load TIFF images
    ds8 = gdal.Open(path+'8band/'+'8band_'+image_id+'.tif')
    ds3 = gdal.Open(path+'3band/'+'3band_'+image_id+'.tif')
    #geoTrans = ds8.GetGeoTransform()
    dist = np.zeros((ds8.RasterXSize, ds8.RasterYSize))

    for i in range(ds8.RasterXSize):
        for j in range(ds8.RasterYSize):
            point = Point(i, j)
            pd = -100000.0
            polygon = shape(record['poly'])
            newpd = point.distance(polygon.boundary)
            if False == polygon.contains(point):
                newpd = -1.0 * newpd
            if newpd > pd:
                pd = newpd
            dist[i,j] = pd

    np.save(path+'CosmiQ_distance/'+fn+'.distance',dist)
