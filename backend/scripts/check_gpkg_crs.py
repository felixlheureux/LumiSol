import geopandas as gpd

path = "quebec_lots.gpkg"
try:
    # Read first row to check CRS
    gdf = gpd.read_file(path, rows=1)
    print(f"CRS: {gdf.crs}")
    print(f"Geometry Type: {gdf.geometry.type.iloc[0]}")
    print(f"Columns: {gdf.columns}")
except Exception as e:
    print(f"Error: {e}")
