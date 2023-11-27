import geopandas as gpd
from shapely.geometry import box


# Đường dẫn đến shapefile gốc
shapefile_path = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/ShipDetection/Ship.shp"
# Đọc shapefile gốc
gdf = gpd.read_file(shapefile_path)

# Tạo một GeoDataFrame mới để lưu bounding box
bbox_gdf = gpd.GeoDataFrame()


# Lặp qua từng polygon trong GeoDataFrame gốc
for index, row in gdf.iterrows():
    polygon = row['geometry']
    bbox = polygon.bounds  # Lấy hộp bao (bounding box) của đa giác
    bbox_polygon = box(*bbox)  # Tạo đa giác từ hộp bao
    bbox_gdf = bbox_gdf.append({'geometry': bbox_polygon}, ignore_index=True)  # Thêm đa giác vào GeoDataFrame mới


# Đường dẫn đến shapefile mới
output_path = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/ShipDetection/Ship_bbox.shp"

# Xuất shapefile mới
bbox_gdf.to_file(output_path)







