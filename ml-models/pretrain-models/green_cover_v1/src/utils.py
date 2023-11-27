import rasterio
import numpy as np

def get_range_value(img):
    data = np.empty(img.shape)
    for i in range(4):
        data[i] = img[i]/15000
        mask = (data[i] <= 1)
        data[i] = data[i]*mask
    return data

def write_color_img(img_path, out_path):
    with rasterio.Env():   
        with rasterio.open(img_path) as src:
            final_img = src.read(1)  
            meta = src.meta.copy()
            meta.update({'nodata': 0, 'dtype':'uint8'})
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(final_img, indexes=1)
            dst.update_tags(AREA_OR_POINT="green='1'")
            dst.write_colormap(1, {
                    0: (0,0,0, 0), 
                    1: (31,255,15,0)})