import os
from src.gendata import sort_path
from src.predict_fast import predict_big
from params import FIRST_IMAGE_PATH, INPUT_PATH, INPUT_PATHS, OUTPUT_PATH, SORT_AMOUNT_OF_CLOUDS, TMP_PATH


def main(list_fp_img_selected, tmp_dir, sort_amount_of_clouds, first_image):
    out_dir_cloud_nodata = os.path.join(tmp_dir, "cloud_nodata")
    if not os.path.exists(out_dir_cloud_nodata):
        os.makedirs(out_dir_cloud_nodata)
    for fp in list_fp_img_selected:
        fn = os.path.basename(fp)
        fp_out = os.path.join(out_dir_cloud_nodata, fn)
        predict_big(fp, fp_out)
    list_name_cloud_nodata = sort_path(
        list_fp_img_selected, out_dir_cloud_nodata, sort_amount_of_clouds, first_image)
    if INPUT_PATH:
        os.rename(f"{out_dir_cloud_nodata}/input.tif", OUTPUT_PATH)
    return out_dir_cloud_nodata, list_name_cloud_nodata


if __name__ == "__main__":
    list_fp_img_selected = INPUT_PATHS
    tmp_dir = TMP_PATH
    out_fp_cloud_remove = OUTPUT_PATH
    sort_amount_of_clouds = SORT_AMOUNT_OF_CLOUDS
    first_image = FIRST_IMAGE_PATH
    input_path = INPUT_PATH
    if input_path:
        list_fp_img_selected = [input_path]
        first_image = input_path
    main(list_fp_img_selected, tmp_dir, sort_amount_of_clouds, first_image)
