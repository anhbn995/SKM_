import os
import argparse
from src import detect_green, detect_water

from src.change_model import get_model
from src.merge_water_green import combine_all
from params import DIL_RESULT, INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, RUN_AGAIN, THRESH_HOLD_GREEN, THRESH_HOLD_WATER, TMP_PATH


def run_segmentation(img, result_path, weight_path_green, weight_path_water, thresh_hold_green, thresh_hold_water,
                     json_path, dil=False, run_again=True):
    green_model, input_size_green = get_model("Green_model", json_path)
    water_model, input_size_water = get_model("Water_model", json_path)
    green_results = os.path.join(
        result_path, os.path.basename(img).replace('.tif', '_green.tif'))
    if not os.path.exists(green_results):
        tmp_green = detect_green.predict(img, result_path, weight_path_green, green_model, input_size_green, dil=dil,
                                         thresh_hold=thresh_hold_green)
    else:
        if run_again:
            tmp_green = detect_green.predict(img, result_path, weight_path_green, green_model, input_size_green,
                                             dil=dil, thresh_hold=thresh_hold_green)
        else:
            tmp_green = None
            pass

    water_results = os.path.join(
        result_path, os.path.basename(img).replace('.tif', '_water.tif'))
    if not os.path.exists(water_results):
        tmp_water = detect_water.predict(img, result_path, weight_path_water, water_model, input_size_water,
                                         thresh_hold=thresh_hold_water)
    else:
        if run_again:
            tmp_water = detect_water.predict(img, result_path, weight_path_water, water_model, input_size_water,
                                             thresh_hold=thresh_hold_water)
        else:
            tmp_water = None
            pass
    return tmp_green, tmp_water


def main(input_path, tmp_path, weight_path_green, weight_path_water, thresh_hold_green, thresh_hold_water, json_path,
         output_path, dil, run_again):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    result_green, result_water = run_segmentation(input_path, tmp_path, weight_path_green, weight_path_water,
                                                  thresh_hold_green, thresh_hold_water, json_path, dil, run_again)
    combine_all(input_path, tmp_path, result_green, result_water, output_path)


if __name__ == "__main__":
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    tmp_path = TMP_PATH
    weight_path_green = f'{ROOT_DATA_FOLDER}/pretrain-models/green-cover/v1/weights/green_weights.h5'
    weight_path_water = f'{ROOT_DATA_FOLDER}/pretrain-models/green-cover/v1/weights/water_weights.h5'
    dil = DIL_RESULT
    run_again = RUN_AGAIN
    thresh_hold_green = THRESH_HOLD_GREEN
    thresh_hold_water = THRESH_HOLD_WATER
    json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/green-cover/v1/model.json'
    main(input_path, tmp_path, weight_path_green, weight_path_water, thresh_hold_green, thresh_hold_water, json_path,
         output_path,
         dil=dil, run_again=run_again)
    print("Finished")
    import sys
    sys.exit()
