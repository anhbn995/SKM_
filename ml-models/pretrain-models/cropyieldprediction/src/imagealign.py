'''
    Align an input image to a reference one for comparing pixel per pixel
'''
#!/usr/bin/env python
from osgeo import gdal, gdalconst
import sys

def align(input, reference, output):
    # Source
    src_filename = input
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    nodata=src.GetRasterBand(1).GetNoDataValue()

    # We want a section of source that matches this:
    match_filename = reference
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = output
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, src.RasterCount, gdalconst.GDT_UInt16)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    dst.GetRasterBand(1).SetNoDataValue(nodata)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

    del dst # Flush
    src=None
    match_ds=None
    dst=None

if __name__=='__main__':
    # input=r"F:\data2019\SAR\QuangNam\s1_07_byte.tif"
    # reference=r"F:\data2019\SAR\QuangNam\s1_09_byte.tif"
    # output=r"F:\data2019\SAR\QuangNam\s1_07_byte_align.tif"
    if len(sys.argv)==4:
        input=sys.argv[1]
        reference=sys.argv[2]
        output=sys.argv[3]
    else:
        print('using imagealign <in> <ref> <out>')
        sys.exit(1)

    print("Align {} to {} and write to {} ...".format(input, reference, output))
    align(input, reference, output)
    print('Finish!')