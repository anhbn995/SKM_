from params import MODEL_META, INPUT_PATH, OUTPUT_PATH, MODEL_PATH, MODEL_TYPE
from predict import UnetPredictor
from osgeo import gdal
import json


def hex_2_rgb(hex):
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, int(hlen // 3)))


if __name__ == "__main__":
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    model_path = MODEL_PATH
    unet_type = MODEL_TYPE
    model_meta = json.loads(MODEL_META)
    for label in model_meta.get('labels'):
        if 'rgb' in label['color']:
            label['color'] ='#' +str('%02x%02x%02x' % tuple(json.loads(label['color'].replace('rgb','').replace('(','[').replace(')',']'))))
    unet_predictor = UnetPredictor(
        input_path,
        model_path,
        output_path,
        unet_type,
        **model_meta
    )
    unet_predictor.predict()
    unet_predictor = None
    ds = gdal.Open(output_path, 1)
    band = ds.GetRasterBand(1)
    colors = gdal.ColorTable()
    for label in model_meta.get('labels'):
        colors.SetColorEntry(label['value'], hex_2_rgb(label['color'][1:]))
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    del band, ds
