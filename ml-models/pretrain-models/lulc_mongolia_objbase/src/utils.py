import rasterio
import numpy as np


def get_quantile_schema(img):
    qt_scheme = []
    with rasterio.open(img) as r:
        num_band = r.count
        for chanel in range(1,num_band+1):
            data = r.read(chanel).astype(np.float16)
            data[data==0] = np.nan
            qt_scheme.append({
                'p2': np.nanpercentile(data, 2),
                'p98': np.nanpercentile(data, 98),
            })
        r.close()

    return qt_scheme
    