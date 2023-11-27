from osgeo import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import os
import random
from tqdm import *
import rasterio
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
        # self.outputFolder_image = os.path.join(self.outputFolder,"image")
        self.outputFolder_label = os.path.join(self.outputFolder,"label")
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if not os.path.exists(self.outputFolder_base):
            os.makedirs(self.outputFolder_base)
        # if not os.path.exists(self.outputFolder_image):
        #     os.makedirs(self.outputFolder_image)
        if not os.path.exists(self.outputFolder_label):
            os.makedirs(self.outputFolder_label)
        base=gdal.Open(self.basefile, GA_ReadOnly)
        # basedata=np.array(base.ReadAsArray())

        # image=gdal.Open(self.imagefile, GA_ReadOnly)
        # imagedata=np.array(image.ReadAsArray())

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
                # lable=rband[py:py+self.sampleSize, px:px+self.sampleSize]
                if np.amax(rband)>0 and np.count_nonzero(rband)>0.005*self.sampleSize*self.sampleSize:
                    geo1=list(geo)
                    geo1[0]=geo[0]+geo[1]*px
                    geo1[3]=geo[3]+geo[5]*py
                    basefile_cr=os.path.join(self.outputFolder_base, self.fileprefix+'_{:03d}.tif'.format(icount+1))
                    gdal.Translate(basefile_cr, base,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    labelfile_cr=os.path.join(self.outputFolder_label, self.fileprefix+'_{:03d}.tif'.format(icount+1))
                    gdal.Translate(labelfile_cr, raster,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    icount+=1
                    pbar.update()


        raster=None
        image=None
        base=None

    def writeLabelAsFile(self, data, filename, geo, proj):
        size_Y, size_X=data.shape
        # Create tiff file
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
        # Create tiff file
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
from get_image_resolution_meter import get_resolution_meter

def main_gen_data_with_size(base_path,mask_path,outputFolder,sampleSize=512):
    list_id = create_list_id(base_path)
    print(list_id)
    for image_id in list_id:
        basefile=os.path.join(base_path,image_id+".tif")
        # imagefile=os.path.join(image_path,image_id+".tif")
        labelfile=os.path.join(mask_path,image_id+".tif")
        resolution = get_resolution_meter(basefile)
        print(image_id,resolution)
        sampleSize = round(0.3*256/(resolution*64))*64
        # sampleSize=1024
        with rasterio.open(labelfile) as src:
            w,h = src.width,src.height
        numgen = w*h//((sampleSize//2)**2)*2
        if numgen >=500:
            numgen = 500
        if numgen <=100:
            numgen = 100
        print(numgen)
        fileprefix = image_id
        gen=CD_GenerateTrainingDataset(basefile, labelfile, sampleSize, outputFolder, fileprefix)
        gen.generateTrainingDataset(numgen)
    return gen.outputFolder_base, gen.outputFolder_label

if __name__=='__main__':
    base_path = "/media/skymap/Backup/indo_sinamas_month6/tmp/image_cut_img"
    # image_path = "/mnt/D850AAB650AA9AB0/changedetection/SLA/image"
    mask_path = "/media/skymap/Backup/indo_sinamas_month6/tmp/image_cut_img_mask"
    outputFolder='/media/skymap/Backup/indo_sinamas_month6/tmp/data_23_06_2020'
    list_id = create_list_id(base_path)
    for image_id in list_id:
        basefile=os.path.join(base_path,image_id+".tif")
        # imagefile=os.path.join(image_path,image_id+".tif")
        labelfile=os.path.join(mask_path,image_id+".tif")
        sampleSize=512
        with rasterio.open(labelfile) as src:
            w,h = src.width,src.height
        numgen = (w*h//((sampleSize//4)**2))
        fileprefix = image_id
        gen=CD_GenerateTrainingDataset(basefile, labelfile, sampleSize, outputFolder, fileprefix)
        gen.generateTrainingDataset(numgen)
