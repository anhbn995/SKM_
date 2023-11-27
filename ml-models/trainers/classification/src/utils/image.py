from rio_tiler.io import COGReader
import rasterio
import numpy as np
from osgeo import gdal, gdalconst
from pyproj import Proj, transform


def get_quantile_schema(img):
    try:
        qt_scheme = []
        with COGReader(img) as cog:
            try:
                stats = cog.stats()
                for _, value in stats.items():
                    qt_scheme.append({
                        'p2': value['pc'][0],
                        'p98': value['pc'][1],
                    })
            except:
                stats = cog.statistics()
                for _, value in stats.items():
                    qt_scheme.append({
                        'p2': value['percentile_2'],
                        'p98': value['percentile_98'],
                    })
        return qt_scheme
    except:
        qt_scheme = []
        with rasterio.open(img) as r:
            num_band = r.count
            for chanel in range(1, num_band + 1):
                data = r.read(chanel).astype(np.float16)
                data[data == 0] = np.nan
                qt_scheme.append({
                    'p2': np.nanpercentile(data, 2),
                    'p98': np.nanpercentile(data, 98),
                })
        return qt_scheme


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


def align_images(input, reference, output):
    # Source
    src_filename = input
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    dtype = src.GetRasterBand(1).DataType

    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    match_filename = reference
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = output
    dst = gdal.GetDriverByName('GTiff').Create(
        dst_filename, wide, high, src.RasterCount, dtype)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

    del dst  # Flush
    src = None
    match_ds = None
    dst = None


def resize_with_resolution_in_meter(in_path, out_path, resolution):
    in_proj = Proj(init='epsg:3857')
    out_proj = Proj(init='epsg:4326')
    x1, y1 = resolution, resolution
    x2, y2 = transform(in_proj, out_proj, x1, y1)

    gdal.Warp(out_path, in_path,
              format='GTiff',
              xRes=x2, yRes=y2,
              resampleAlg=gdal.GRA_Average)
