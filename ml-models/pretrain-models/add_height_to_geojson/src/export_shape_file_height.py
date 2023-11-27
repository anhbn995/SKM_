import fiona
from rasterstats import zonal_stats

 
def export_shp_height(in_img_height, in_shp_building, out_shape_height_building):
    with fiona.open(in_shp_building) as src:
        zs = zonal_stats(src, in_img_height, stats='median', nodata=-32768)
        size=len(zs)
        schema=src.schema
        schema['properties']['height'] = 'float'
        
        with fiona.open(out_shape_height_building, 'w', crs=src.crs, driver=src.driver, schema=schema) as dst:
            for idx, f in enumerate(src):
                f['properties'].update(height=zs[idx]['median'])
                dst.write(f)
                print(f'Feature {idx}/{size}')
                
if __name__=='__main__':
    in_img_height = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE_2/test/01_p5_ds2_rs.tif'
    in_shp_building = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test_new/01_p5_ds2/01_p5_ds2.shp'
    out_shape_height_building = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE_2/test/a.shp'
    export_shp_height(in_img_height, in_shp_building, out_shape_height_building)