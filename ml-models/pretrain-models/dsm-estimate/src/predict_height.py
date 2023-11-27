import torch
from models import modules, net, senet
import rasterio
import numpy as np
import math
from rasterstats import zonal_stats
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def one2two(i, xsize): #convert vector index to matrix index; x - column, y - row
  x= i % xsize
  y= i // xsize
  return (x, y)


def normalize(images, means, stds):
    for t, m, s in zip(images, means, stds):
        t.sub_(m).div_(s)
    return images.float()


def predict_batch(images_inbatch, model, device):
    batch=np.stack(images_inbatch, axis=0)
    
    batch=torch.from_numpy(batch)
    
    batch=normalize(batch, IMAGENET_STATS['mean'], IMAGENET_STATS['std'])
    
    batch=batch.to(device)
    with torch.no_grad():
        pred = model(batch)

    pred = torch.nn.functional.interpolate(pred,size=(440,440),mode='bilinear')
    pred = pred.detach().cpu().numpy()
    pred= pred*100
    
    pred=pred.squeeze()
    return pred


# predict by batchsize
def predict(inputfile, outputfile, model, input_size=440, padding=30, batchsize=2):
    with rasterio.open(inputfile) as src:
        w,h = src.width,src.height
        source=src.read().transpose(1,2,0)[:,:,0:3] #get first 3 bands
        profile = src.profile
    
    # predict
    crop_size=input_size-2*padding

    if h%crop_size==0:
        new_h=h
    else:
        new_h=h+crop_size-(h%crop_size)

    if w%crop_size==0:
        new_w=w
    else:
        new_w=w+crop_size-(w%crop_size)

    top_pad=padding
    bottom_pad=padding + new_h - h
    left_pad=padding
    right_pad=padding + new_w - w
    top_pad, left_pad, bottom_pad, right_pad

    # padding image
    source_pad=np.pad(source, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
    ylist=range(padding, source_pad.shape[0]-2*padding,crop_size)
    xlist=range(padding, source_pad.shape[1]-2*padding,crop_size)

    print('Predicting {} rows of {} columns...'.format(len(ylist), len(xlist)))
    
    outshape_pad=source_pad.shape[:-1]
    farm_pad=np.zeros(outshape_pad, dtype=np.uint8)

    size=len(ylist) * len(xlist)
    index=0
    while index<size:
        print('Batch number: {}/{}'.format(index // batchsize, math.ceil(size / batchsize)))
        arr=[]
        for i in range(batchsize):
            if (index+i<size):
                x, y = one2two(index+i, len(xlist))
                arr.append((ylist[y], xlist[x]))
        index+=batchsize

        images_inbatch=[]
        for (y,x) in arr:
            img=source_pad[y-padding:y+crop_size+padding,x-padding:x+crop_size+padding,:]
            img = np.array(img/255.0, dtype=np.float32).transpose(2,0,1)
            images_inbatch.append(img)

        pred=predict_batch(images_inbatch, model, device)

        for i, (y, x) in enumerate(arr):
            if len(pred.shape)==3: # batch, h, w
                farm_pad[y:y+crop_size,x:x+crop_size]=pred[i][padding:padding+crop_size, padding: padding+crop_size]
            else:
                farm_pad[y:y+crop_size,x:x+crop_size]=pred[padding:padding+crop_size, padding: padding+crop_size]

    # clean
    source_pad=None
    source=None

    # clip to the original shape        
    farm_final=farm_pad[padding:h+padding, padding:w+padding]
    
    # Write out
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    #farm boundary
    with rasterio.open(outputfile, 'w', **profile) as dst:
        dst.write(farm_final, 1)


def loadmodel(modelfile, device):
    
    original_model = senet.senet154(pretrained=None)
    Encoder = modules.E_senet(original_model)
    model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    model.to(device)
    state_dict = torch.load(modelfile)['state_dict']

    del state_dict["E.Harm.dct"]
    del state_dict["E.Harm.weight"]
    del state_dict["E.Harm.bias"]

    model.load_state_dict(state_dict)

    return model


def predict_height(inputfile, outputfile, modelfile):
    model=loadmodel(modelfile, device)
    model.eval()
    predict(inputfile, outputfile, model)
    

if __name__=='__main__':

    modelfile=r"/home/geoai/ml-models/pretrain-models/dsm-estimate/v1/weights/model.pth.tar"
    inputfile=r'/home/geoai/ml-models/pretrain-models/dsm-estimate/v1/example-data/tmp.tif'
    outputfile=r'/home/geoai/ml-models/pretrain-models/dsm-estimate/test/tmp_he.tif'
    predict_height(inputfile, outputfile, modelfile)
