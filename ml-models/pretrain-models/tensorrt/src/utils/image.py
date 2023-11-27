from osgeo import gdal
import numpy as np
import rasterio


def stretch_image(input, output):
    ds = gdal.Open(input, gdal.GA_ReadOnly)
    bcount = ds.RasterCount
    rows = ds.RasterXSize
    cols = ds.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output, rows, cols, bcount, gdal.GDT_Byte)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input

    from rio_tiler.io import COGReader
    with COGReader(input) as cog:
        qt_scheme = []

        stats = cog.stats(pmax=98, pmin=2)
        for _, value in stats.items():
            try:
                qt_scheme.append({
                    'p2': value['pc'][0],
                    'p98': value['pc'][1],
                })
            except:
                qt_scheme.append({
                    'p2': value['percentiles'][0],
                    'p98': value['percentiles'][1],
                })

    for i in range(bcount):
        band = np.array(ds.GetRasterBand(i + 1).ReadAsArray())
        band_qt = qt_scheme[i]
        band = np.interp(
            band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
        outdata.GetRasterBand(i + 1).WriteArray(band)
        outdata.GetRasterBand(i + 1).SetNoDataValue(0)
        band = None
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    # close dataset
    ds = None


def get_info_image(image_path):
    with rasterio.open(image_path) as src:
        crs = dict(src.crs)
        transform = src.transform
        w, h = src.width, src.height
    return h, w, crs, transform
