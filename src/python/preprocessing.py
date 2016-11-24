from spaceNet import geoTools as gT
from shapely.geometry import Polygon, shape, Point
import numpy as np
import sys

# The file name
fn = sys.argv[1]
# The path where TIFF files located
path = sys.argv[2]


def Pixel2World ( geoMatrix, i , j ):
    ```
    Converts pixel coordinates to the world coordinates
    ```
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    return(1.0 * i * xDist  + ulX, -1.0 * j * xDist + ulY)



# read WKT file with buildings footrpints
buildinglist = readwktcsv(path + 'vectorData/wkt/truth.csv')
for record in buildinglist:
    image_id = record['ImageId']
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
