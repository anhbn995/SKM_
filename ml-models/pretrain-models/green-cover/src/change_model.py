import os
import json
import argparse

from src.utils import write_json_file
from src.models.models import *

def get_model(model_name, json_path=None):
    if not json_path:
        json_path = os.path.join('.', 'model.json')
    if os.path.exists(json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
    else:
        raise Exception("Requirement file isn't exists, please check %s"%(json_path))

    model = data[model_name]["name"]
    n_labels = data[model_name]['n_labels']
    input_size = data[model_name]['input_size']
    filter_num = data[model_name]['filter_num']
    if 'batch_norm' in data[model_name]:
        batch_norm = data[model_name]['batch_norm']
    else:
        batch_norm = False

    models = eval(model)
    models = models(input_size=input_size, filter_num=filter_num, n_labels=n_labels, batch_norm=batch_norm)
    return models, input_size

def change_model(type_model, name_model, input_size, filter_num, n_labels):
    json_path = os.path.join(os.getcwd(), 'model.json')
    if os.path.exists(json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
    else:
        raise Exception("Requirement file isn't exists, please check %s"%(json_path))
    
    model = data["%s"%(type_model)]
    if type_model not in ['Cloud_model', 'Green_model', 'Water_model']:
        raise Exception("Type of model isn't in list model %s"%(['Cloud_model', 'Green_model', 'Water_model']))

    if name_model:
        try:
            eval(name_model)
        except:
            raise Exception("%s isn't name of function, or model isn't available"%(model["name"]))
        
        model.update({"name":'%s'%(name_model)})

    if input_size:
        if isinstance(input_size, list):
            if len(input_size)==3:
                if not isinstance(input_size[0], int):
                    raise Exception("Type of W,H,C must be int")
            else:
                raise Exception("Input size must contains w,h,channels")
        else:
            raise Exception("Input size must be list [w,h,channels]")

        model.update({"input_size":input_size})

    if filter_num:
        if isinstance(filter_num, list):
            if not isinstance(filter_num[0], int):
                raise Exception("Type of num_filter must be int")
        else:
            raise Exception("Num_filer must be list filter of each stage")

        model.update({"filter_num":filter_num})

    if n_labels:
        if not isinstance(n_labels, int):
            raise Exception("Type of n_labels must be int")

        model.update({"n_labels":n_labels})

    write_json_file(data, 'model')
    return json_path

if __name__=="__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--type_model', help='Type of model', required=True, type=str,
                            default='Green_model')
    args_parser.add_argument('--name_model', help='Name of model', required=True, type=str,
                            default='att_unet')                        
    args_parser.add_argument('--filter_num', help='List of filers', required=True, 
                            type=str, default='128,256,512,1024')
    args_parser.add_argument('--input_size', help='Size of the input', required=True, 
                            type=str, default='128,128,4')
    args_parser.add_argument('--n_labels', help='Number of the objects', required=False, 
                            type=int, default=1)

    param = args_parser.parse_args()
    n_labels = param.n_labels
    type_model = param.type_model
    name_model = param.name_model
    filter_num = []
    input_size = []
    
    for i in param.input_size.split(','):
        if i == '':
            pass
        input_size.append(int(i))
    
    for j in param.filter_num.split(','):
        if j == '':
            pass
        filter_num.append(int(j))

    change_model(type_model, name_model, input_size, filter_num, n_labels)

    