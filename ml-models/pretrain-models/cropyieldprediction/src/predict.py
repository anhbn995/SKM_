import rasterio
import numpy as np
from tensorflow.keras.models import load_model
import sys, math
from stretch import stretch

# load model
def loadmodel(weightsfile):
    model = load_model(weightsfile)

    return model

def one2two(i, xsize): #convert vector index to matrix index; x - column, y - row
  x= i % xsize
  y= i // xsize
  return (x, y)

# predict by batchsize
def predict(inputfiles, outputfile, model, patchsize=33, batchsize=1000):
    with rasterio.open(inputfiles[0]) as src: # uint8
        w,h = src.width,src.height
        source=src.read().transpose(1,2,0)
        profile = src.profile
        nodata_msk=src.read_masks(1)
        
    sourcestack=[source]
    for i in range(1,len(inputfiles)):
        with rasterio.open(inputfiles[i]) as src: # uint8
            data=src.read().transpose(1,2,0)
            sourcestack.append(data)
            
    source=np.stack(sourcestack, axis=2)
        

#     # Stack or expands time dimension
#     source=np.expand_dims(source, axis=2)
    
    # predict
    padding=math.floor(patchsize/2)

    # padding image
    source_pad=np.pad(source, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

    print('Predicting {} rows of {} columns...'.format(h, w))
    
    outshape_pad=source.shape[:-2]
    farm_pad=np.zeros(outshape_pad, dtype=np.float32)

    size=h * w
    index=0
    while index<size:
        print('Batch number: {}/{}'.format(index // batchsize, math.ceil(size / batchsize)))
        arr=[]
        for i in range(batchsize):
            if (index+i<size):
                x, y = one2two(index+i, w)
                arr.append((y, x))
        index+=batchsize

        images_inbatch=[]
        for (y,x) in arr:
            img=source_pad[y:y+2*padding+1,x:x+2*padding+1,:,:]
            img = np.array(img/255.0, dtype=np.float32)
            images_inbatch.append(img)            

        pred=model.predict(np.stack(images_inbatch, axis=0), verbose=1)

        for i, (y, x) in enumerate(arr):
            farm_pad[y,x]=pred[i]

    # clean
    source_pad=None
    source=None

    
    # Write out
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    # nodata
    farm_pad[nodata_msk==0]=0

    #farm boundary
    with rasterio.open(outputfile, 'w', **profile) as dst:
        dst.write(farm_pad, 1)

if __name__=='__main__':
    if len(sys.argv)<3:
        print("Error! please use: python predict.py <modelfile> <input1> [<input2> [<input3>]]")
        sys.exit(1)
    
    modelfile=sys.argv[1]
    inputfiles=[sys.argv[2]]
    for i in range(3, len(sys.argv)):
        inputfiles.append(sys.argv[i])

    outputfile=inputfiles[-1].replace(".tif","_predict.tif")

    inputfiles_tmp=[f.replace(".tif", "_tmp.tif") for f in inputfiles]
    for i in range(len(inputfiles)):
        stretch(inputfiles[i], inputfiles_tmp[i])

    model=loadmodel(modelfile)

    predict(inputfiles_tmp, outputfile, model)    
