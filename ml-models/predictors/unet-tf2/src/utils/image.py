
import rasterio
import numpy as np

def get_quantile_schema(img):

    # try:
    #     from rio_tiler.io import COGReader
    #     qt_scheme = []
    #     with COGReader(img) as cog:
    #         try:
    #             stats = cog.stats()
    #             for _, value in stats.items():
    #                 qt_scheme.append({
    #                     'p2': value['pc'][0],
    #                     'p98': value['pc'][1],
    #                 })
    #         except:
    #             stats = cog.statistics()
    #             for _, value in stats.items():
    #                 qt_scheme.append({
    #                     'p2': value['percentile_2'],
    #                     'p98': value['percentile_98'],
    #                 })
    #     return qt_scheme
    # except:
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





