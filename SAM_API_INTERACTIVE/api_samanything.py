import rasterio
import numpy as np
from flask import request, Flask
from flask_restful import Resource, Api
from uuid import uuid4
from samgeo import SamGeo
from predict_anything_sam import *
# from predict_point_sam import *
from predict_text_sam import *
import geopandas as gpd

app = Flask(__name__)
api = Api(app)
app_port = 5000

# load model global
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_sam = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/weights/sam_vit_h_4b8939.pth"
model_sam_anything = SamGeo(
    checkpoint=model_path,
    model_type="vit_h",
    automatic=True,
    device=device,
    sam_kwargs=None,
)

# model_type = 'vit_h'
# sam_model_samtext = sam_model_registry[model_type](checkpoint=model_path)
# sam_model_samtext.to(device=device_sam)
# sam_model_samtext = SamPredictor(sam_model_samtext)

# model_diano = GoundingDinoBuildSam()
# model_diano_ = model_diano.build_groundingdino()


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

        
class ImagePredictionSamAnything(Resource):
    def post(self):
        image_path = request.form.get('image_path')
        if os.path.exists(image_path):
            output_path = f'./result/{os.path.basename(image_path).replace(".tif","")}_predict_anything.shp'
            name_folder_tmp = os.path.basename(image_path).replace(".tif","")
            tmp_dir = f'./tmp/{name_folder_tmp}' + uuid4().hex + '_samanything'
            os.makedirs(tmp_dir, exist_ok=True)
            check = main_sam_anything(model_sam_anything, image_path, output_path, tmp_dir, size_img = 1700)
            if check == 'Done':
                gdf = gpd.read_file(output_path)
                return gdf.to_json()
            else:
                return {'message': check}, 200
        else:
            return {'message': 'path image is not define'}, 200



api.add_resource(UploadImage, '/upload')
api.add_resource(ImagePredictionSamAnything, '/predict_anything')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=app_port)
        
    