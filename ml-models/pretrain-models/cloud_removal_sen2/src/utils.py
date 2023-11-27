import numpy as np

def get_normalized_data(data_image, data_type, clip_min=None, clip_max=None):
    scale = 2000
    max_val_opt = 10000
    max_val_sar = 2
    clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # clip_min = [[-32.5, -25.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    clip_max = [[0, 0], [max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt],
                [max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt, max_val_opt]]
    shift_data = False

    shift_values = [[0, 0], [1300., 981., 810., 380., 990., 2270., 2070., 2140., 2200., 650., 15., 1600., 680.],
                    [1545., 1212., 1012., 713., 1212., 2476., 2842., 2775., 3174., 546., 24., 1776., 813.]]

    # SAR
    if data_type == 1:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                            clip_max[data_type - 1][channel])
            data_image[channel] -= clip_min[data_type - 1][channel]
            data_image[channel] = max_val_sar * (data_image[channel] / (
                    clip_max[data_type - 1][channel] - clip_min[data_type - 1][channel]))
        if shift_data:
            data_image -= max_val_sar / 2
    # OPT
    elif data_type == 2 or data_type == 3:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                            clip_max[data_type - 1][channel])
            if shift_data:
                data_image[channel] -= shift_values[data_type - 1][channel]

        data_image /= scale

    return data_image