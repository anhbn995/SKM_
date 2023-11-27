import rasterio
import numpy as np
from uuid import uuid4
from predict_text_sam import *
import geopandas as gpd
import pandas as pd

from flask import request, Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
app_port = 5000

# load model global
device_sam = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth"
model_type = 'vit_h'
sam_model_samtext = sam_model_registry[model_type](checkpoint=model_path)
sam_model_samtext.to(device=device_sam)
sam_model_samtext = SamPredictor(sam_model_samtext)

model_diano = GoundingDinoBuildSam()
model_diano_ = model_diano.build_groundingdino()
print('Done'*100)


class UploadImage(Resource):
    def post(self):
        # Upload file
        # Kiểm tra xem tệp đã được gửi lên hay chưa
        if 'image' not in request.files:
            return {'message': 'Không có tệp nào được gửi lên'}, 400

        image_file = request.files['image']
        
        # Kiểm tra định dạng tệp
        if image_file.filename == '':
            return {'message': 'Không có tên tệp'}, 400
        if not image_file.filename.endswith('.tif'):
            return {'message': 'Tệp phải có định dạng .tif'}, 400

        # Lưu tệp vào thư mục upload
        img_name = uuid4().hex
        image_path = f'./upload/{img_name}.tif'
        image_file.save(image_path)
        return {'message': 'Tải lên thành công', 'image_path': image_path}, 200


class ImagePredictionSamText(Resource):
    def post(self):
        data = request.get_json(force=True)
        image_path = data['image_path']
        text_prompts = data['text_prompt']
        if os.path.exists(image_path):
            if len(text_prompts) < 1:
                return {'message': 'send prompt text empty'}
            else:
                check = 0
                for text_prompt in text_prompts:
                    print(text_prompt, 'run sam')
                    name_folder_tmp = os.path.basename(image_path).replace(".tif","")
                    tmp_dir = f'./tmp/{name_folder_tmp}' + uuid4().hex + f'_samtext_{text_prompt}'
                    os.makedirs(tmp_dir, exist_ok=True)
                    output_path = f'./result/{os.path.basename(image_path).replace(".tif","")}_predict_samtext_{text_prompt}.shp'
                    count = main_sam_text(model_diano_, sam_model_samtext, image_path, text_prompt, tmp_dir, output_path)
                    check = check + count
                    print(count, 'x'*100)
                
                if check < 1:
                    return {'message': 'no object'}
                else:
                    gdf_rs = []
                    for text_prompt in text_prompts:
                        tmp_shp = f'./result/{os.path.basename(image_path).replace(".tif","")}_predict_samtext_{text_prompt}.shp'
                        if os.path.exists(tmp_shp):
                            print(text_prompt)
                            gdf_them = gpd.read_file(tmp_shp)
                            gdf_them['class'] = text_prompt
                            gdf_rs.append(gdf_them)
                    gdf_rs = gpd.GeoDataFrame(pd.concat(gdf_rs, ignore_index=True))
                    x = f'./result/{os.path.basename(image_path).replace(".tif","")}_predict_samtext_aaaaaaaaaa.shp'
                    gdf_rs.to_file(x)
                    return gdf_rs.to_json()
        else:
            return {'message': 'path image is not define'}, 200
        
api.add_resource(UploadImage, '/upload')
api.add_resource(ImagePredictionSamText, '/predict_text')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0",port=app_port)