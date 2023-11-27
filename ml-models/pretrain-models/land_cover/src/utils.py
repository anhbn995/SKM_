import fiona
import rasterio
from rasterio import features as rio_features
import numpy as np
import fnmatch
import shapely
import os
from crop_data_with_size import CD_GenerateTrainingDataset, create_list_id


def _generate_maskfile(imagefile, labelshapefile, maskfile):

    with fiona.open(labelshapefile, "r") as shapefile:
        labels = [feature["geometry"] for feature in shapefile]
    
    with rasterio.open(imagefile) as src:
        height = src.height
        width = src.width
        src_transform = src.transform
        out_meta = src.meta.copy()
        out_meta['dtype']='uint8'
        out_meta.update({'count':1})
        nodatamask = src.read_masks(1) == 0

    mask = rio_features.geometry_mask(
        labels,
        (height, width),
        src_transform,
        all_touched=True,
        invert=True
    ).astype(np.uint8)

    print(np.unique(mask))
    mask[nodatamask] = 0

    # id_mat = np.ones_like(mask)
    # mask = mask + id_mat

    with rasterio.open(maskfile, "w", **out_meta) as dest:
        dest.write(mask, indexes=1)

def crop_raster_by_shape(image_path, shape_path, result_dir):

    #usage: crop raster by box
    #image and shape have same CRS
    print(image_path, shape_path)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    num_file = len(fnmatch.filter(os.listdir(result_dir), '*.tif'))
    path_result = result_dir + '/{}.tif'
    with rasterio.open(image_path, 'r', driver='GTiff') as src:

        index = 1
        for feat in fiona.open(shape_path):
            polygon_geometry = feat['geometry']
            if polygon_geometry['type'] == 'Polygon':
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0])
            else:
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0][0])
            from rasterio.coords import disjoint_bounds
            polygon_bounds = polygon.bounds
            if disjoint_bounds(polygon_bounds, src.bounds):
                raise Exception(
                    'Polygon must overlap the extent of the input raster')
            bounds_window = src.window(*polygon_bounds)

            out_window = bounds_window.round_lengths()
            height = int(out_window.height)
            width = int(out_window.width)
            out_kwargs = src.profile
            out_kwargs.update({
                'height': height,
                'driver': 'GTiff',
                'width': width,
                'transform': src.window_transform(out_window)})

            with rasterio.open(path_result.format(num_file + index), 'w', **out_kwargs) as out:
                out.write(
                    src.read(
                        window=out_window,
                        out_shape=(src.count, height, width),
                        boundless=True,
                        masked=True,
                    )
                )

            index += 1

def gen_trainning_dataset(image_crop_dir, mask_crop_dir, output_dir):
    list_id = create_list_id(image_crop_dir)
    sampleSize=256
    for image_id in list_id:
        basefile=os.path.join(image_crop_dir,image_id+".tif")
        labelfile=os.path.join(mask_crop_dir,image_id+".tif")
        
        with rasterio.open(labelfile) as src:
            w,h = src.width,src.height
        
        if w < sampleSize or h < sampleSize:
            continue
        
        # numgen = (w*h//((sampleSize//4)**2))
        numgen = (w*h)//(sampleSize**2)
        fileprefix = image_id
        try:
            gen=CD_GenerateTrainingDataset(basefile, labelfile, sampleSize, output_dir, fileprefix)
            gen.generateTrainingDataset(numgen)
        except:
            pass
    return True

def get_range_value(img, val=15000):
    data = np.empty(img.shape)
    for i in range(4):
        data[i] = img[i]/val
        mask = (data[i] <= 1)
        data[i] = data[i]*mask
    return data

def get_data(path_train, size=128, numbands=4, train=True):
    # label_count=len(labels) + 1
    ids = next(os.walk(path_train + "/images"))[2]
    X = np.zeros((len(ids), size, size, numbands), dtype=np.float32)
    if train:
        # y = np.zeros((len(ids), size, size, label_count), dtype=np.float32)
        y = np.zeros((len(ids), size, size, 1), dtype=np.float32)
    print('Getting images ... ')

    for n, id_ in enumerate(ids):
        # Load images
        x_img = rasterio.open(path_train + '/images/' + id_).read()
        x_img = get_range_value(x_img)
        x_img = np.transpose(x_img, (1,2,0))
        # Load masks
        if train:
                mask = rasterio.open(path_train + '/masks/' + id_).read()
                onehot = np.zeros((size,size,1))
                onehot[..., 0] = (mask==1).astype(np.uint8)
                            
        # Save images
        try:
            X[n, ...] = x_img.squeeze()
        except:
            X[n, ...] = x_img
        
        if train:
            try:
                y[n,...] = onehot.squeeze().astype(np.uint8)
            except:
                y[n, ...] = onehot.astype(np.uint8)
            
    print('Done!')
    if train:
        return X, y
    else:
        return X

# if __name__=="__main__":
#     # step1: gen mask file
#     # imagefile = '/home/boom/boom/segmantic_segmentation/images/SanJose_2022.tif'
#     # labelshapefile = '/home/boom/boom/segmantic_segmentation/labels/SanJose_2022_shape.shp'
#     # maskfile = '/home/boom/boom/segmantic_segmentation/tmp/masks/SanJose_2022.tif'
#     # _generate_maskfile(imagefile, labelshapefile, maskfile)

#     #step2: crop mask and image with box
#     # raster_path = '/home/boom/boom/segmantic_segmentation/tmp/masks/SanJose_2022.tif'
#     # box_path = '/home/boom/boom/segmantic_segmentation/box/SanJose_2022_box.shp'
#     # result_dir = '/home/boom/boom/segmantic_segmentation/tmp/mask_crop'
#     # crop_raster_by_shape(raster_path, box_path, result_dir)

#     #step3: generate trainning dataset
#     image_crop_dir = '/home/boom/boom/segmantic_segmentation/tmp/image_crop'
#     mask_crop_dir = '/home/boom/boom/segmantic_segmentation/tmp/mask_crop'
#     output_dir = '/home/boom/boom/segmantic_segmentation/tmp/training_dataset'
#     gen_trainning_dataset(image_crop_dir, mask_crop_dir, output_dir)
