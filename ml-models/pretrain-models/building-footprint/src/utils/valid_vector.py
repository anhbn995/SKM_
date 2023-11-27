from shapely.geometry import Polygon, MultiPolygon


def remove_invalid_gdf(gdf):
    for ind, g in enumerate(gdf['geometry']):
        if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
            gdf = gdf.drop(ind)
    return gdf
