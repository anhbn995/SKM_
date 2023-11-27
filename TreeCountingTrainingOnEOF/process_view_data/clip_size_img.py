import os
import uuid
import random
import rasterio
import numpy as np

from tqdm import *
from osgeo import gdal
from gdalconst import GA_ReadOnly
from get_resolution import get_resolution_meter


class CD_GenerateTrainingDataset:
    def __init__(self, basefile, labelfile, sampleSize, outputFolder, fileprefix):
        self.basefile=basefile
        # self.imagefile=imagefile
        self.labelfile=labelfile
        self.sampleSize=sampleSize
        self.outputFolder=outputFolder
        self.fileprefix=fileprefix
        self.outputFolder_base=None
        self.outputFolder_image=None
        self.outputFolder_label = None


    def generateTrainingDataset(self, nSamples):
        self.outputFolder_base = os.path.join(self.outputFolder,"image")
        self.outputFolder_label = os.path.join(self.outputFolder,"label")
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder, exist_ok=True)
        if not os.path.exists(self.outputFolder_base):
            os.makedirs(self.outputFolder_base)
        if not os.path.exists(self.outputFolder_label):
            os.makedirs(self.outputFolder_label)
        base=gdal.Open(self.basefile, GA_ReadOnly)

        raster = gdal.Open(self.labelfile, GA_ReadOnly)
        geo = raster.GetGeoTransform()
        proj=raster.GetProjectionRef()
        size_X=raster.RasterXSize
        size_Y=raster.RasterYSize

        rband=np.array(raster.GetRasterBand(1).ReadAsArray())

        icount=0
        with tqdm(total=nSamples) as pbar:
            while icount<nSamples:
                px=random.randint(0,size_X-1-self.sampleSize)
                py=random.randint(0,size_Y-1-self.sampleSize)
                rband = raster.GetRasterBand(1).ReadAsArray(px, py, self.sampleSize, self.sampleSize)
                if np.amax(rband)>0 and np.count_nonzero(rband)>0.005*self.sampleSize*self.sampleSize:
                    name_file = uuid.uuid4().hex
                    geo1=list(geo)
                    geo1[0]=geo[0]+geo[1]*px
                    geo1[3]=geo[3]+geo[5]*py
                    basefile_cr=os.path.join(self.outputFolder_base, self.fileprefix + f'_{name_file}.tif')
                    gdal.Translate(basefile_cr, base,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    labelfile_cr=os.path.join(self.outputFolder_label, self.fileprefix + f'_{name_file}.tif')
                    gdal.Translate(labelfile_cr, raster,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    icount+=1
                    pbar.update()
        raster=None
        image=None
        base=None


    def writeLabelAsFile(self, data, filename, geo, proj):
        size_Y, size_X=data.shape
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        band = target_ds.GetRasterBand(1)
        target_ds.GetRasterBand(1).SetNoDataValue(0)				
        band.WriteArray(data)
        band.FlushCache()
        target_ds=None
        
        
    def writeDataAsFile(self, data, filename, geo, proj):
        nbands, size_Y, size_X=data.shape
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, nbands, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        for i in range(0, nbands):
            band = target_ds.GetRasterBand(i+1)
            band.SetNoDataValue(0)	
            band.WriteArray(data[i,:,:])
            band.FlushCache()
        target_ds=None    
            
            
def create_list_id(path):
    list_id = []
    for file in os.listdir(path):
        if file.endswith(".tif"):
            list_id.append(file[:-4])
    return list_id


def main_gen_data_with_size(dir_img_aoi, dir_mask_aoi, outputFolder, sampleSize=None, gen_them=False):
    list_id = create_list_id(dir_img_aoi)
    print(list_id)
    for image_id in list_id:
        basefile=os.path.join(dir_img_aoi, image_id+".tif")
        labelfile=os.path.join(dir_mask_aoi, image_id+".tif")
        resolution = get_resolution_meter(basefile)
        print(image_id,resolution)
        if sampleSize:
            size_cut = sampleSize
        else:
            size_cut = round(0.3*256/(resolution*64))*64
        print(size_cut)
        with rasterio.open(labelfile) as src:
            w,h = src.width,src.height
        numgen = w*h//((size_cut//2)**2)*2
        # print(numgen,'z')
        if gen_them:
            numgen = 200
        # print(numgen)
        fileprefix = image_id
        gen=CD_GenerateTrainingDataset(basefile, labelfile, size_cut, outputFolder, fileprefix)
        gen.generateTrainingDataset(numgen)
    return gen.outputFolder_base, gen.outputFolder_label