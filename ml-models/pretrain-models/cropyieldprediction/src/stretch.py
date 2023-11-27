'''
    Convert Uint16 to Uint8 image, nodata=0
'''
from osgeo import gdal
import sys
import numpy as np

def stretch(input, output, mode=1):
    print('Stretching...')
    # open dataset
    print(input)
    ds = gdal.Open(input, gdal.GA_ReadOnly)
    bcount=ds.RasterCount
    rows = ds.RasterXSize
    cols = ds.RasterYSize    

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output, rows, cols, bcount, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input

    for i in range (bcount):
        band = np.array(ds.GetRasterBand(i+1).ReadAsArray())
        nodatamask=(band==0)
        band1=band.astype(float)
        band1[nodatamask]=np.nan
        if mode==1: #cumulative
            p2 = np.nanpercentile(band1, 2)
            p98 = np.nanpercentile(band1, 98)
        else: #standard deviation
            mean=np.nanmean(band1)
            std=np.nanstd(band1)
            p2=mean-std*2
            p98=mean+std*2

        band1=None

        print ("{}, {}".format(p2, p98))
        band=np.interp(band, (p2, p98), (1, 255)).astype(int)
        band[nodatamask]=0
        outdata.GetRasterBand(i+1).WriteArray(band)
        outdata.GetRasterBand(i+1).SetNoDataValue(0)
        band=None

    outdata.FlushCache() ##saves to disk!!
    outdata = None

    # close dataset
    ds = None  

if __name__=='__main__':
    if len(sys.argv)!=3:
        print("Error! please use: python stretch.py <inputimage> <outputimage>")
        sys.exit(1)
    
    imagefile=sys.argv[1]
    filename=sys.argv[2]
    
    stretch(imagefile, filename)