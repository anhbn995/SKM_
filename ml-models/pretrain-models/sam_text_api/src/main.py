from uuid import uuid4
from predict_sam_text import *
import geopandas as gpd
import pandas as pd

from flask import request, Flask,send_file
from flask_restful import Resource, Api
import params


app = Flask(__name__)
api = Api(app)
app_port = 5000

# load model global
device_sam = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"/home/ubuntu/weights/sam_vit_b_01ec64.pth"
model_type = 'vit_b'
sam_model_samtext = sam_model_registry[model_type](checkpoint=model_path)
sam_model_samtext.to(device=device_sam)
sam_model_predict = SamPredictor(sam_model_samtext)

model_diano = GoundingDinoBuildSam()
model_diano_ = model_diano.build_groundingdino()
print('Done ...')

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
        jobid = uuid4().hex
        parent = os.path.dirname(image_path)
        temp_dir = os.path.join(parent,jobid)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        if len(text_prompts) < 1:
            return {'message': 'send prompt text empty'}
        else:
            check = 0
            for text_prompt in text_prompts:
                print(text_prompt, 'run sam')
                output_path = os.path.join(temp_dir,f'{text_prompt}.shp')
                batch_size=2
                count = main_sam_text(model_diano_, sam_model_predict, image_path, text_prompt, temp_dir, output_path, batch_size, sam_model_samtext)
                check = check + count
                print(count, 'x'*100)
            
            if check < 1:
                return {'message': 'no object'}
            else:
                gdf_rs = []
                for text_prompt in text_prompts:
                    output_path = os.path.join(temp_dir,f'{text_prompt}.shp')
                    if os.path.exists(output_path):
                        print(text_prompt)
                        gdf_them = gpd.read_file(output_path)
                        gdf_them['class'] = text_prompt
                        gdf_rs.append(gdf_them)
                gdf_rs = gpd.GeoDataFrame(pd.concat(gdf_rs, ignore_index=True))
                x = os.path.join(temp_dir, f'{jobid}.geojson')
                gdf_rs.to_file(x,driver="GeoJSON")
                return send_file(x)

        
api.add_resource(UploadImage, '/sam/upload')
api.add_resource(ImagePredictionSamText, '/sam/predict_text')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0",port=app_port)