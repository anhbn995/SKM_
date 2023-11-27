import rasterio
import numpy as np
from uuid import uuid4

from predict_point_sam import *

from flask import request, Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
app_port = 6000

# load model global

model_path = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth"
model_sam_point = SamGeo(
    checkpoint=model_path,
    model_type="vit_h", # vit_l, vit_b , vit_h
    automatic=False,
    device=device,
    sam_kwargs=None,
)


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


class ImagePredictionSamPoint(Resource):
    def post(self):
        image_path = request.form.get('image_path')
        input_point_prompt_shp = request.form.get('shp_prompt')

        if os.path.exists(image_path):
            output_path = f'./result/{os.path.basename(image_path).replace(".tif","")}_predict_sampoint.shp'
            name_folder_tmp = os.path.basename(image_path).replace(".tif","")
            tmp_dir = f'./tmp/{name_folder_tmp}' + uuid4().hex + '_sampoint'
            os.makedirs(tmp_dir, exist_ok=True)
            print(input_point_prompt_shp)
            try:
                sam_point_main(model_sam_point, image_path, input_point_prompt_shp, output_path, tmp_dir)
                gdf = gpd.read_file(output_path)
                return gdf.to_json()
            except:
                return {'message': 'done match shape with image'}, 200
        else:
            return {'message': 'path image is not define'}, 200
        
        
        
api.add_resource(UploadImage, '/upload')
api.add_resource(ImagePredictionSamPoint, '/predict_point')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0",port=app_port)