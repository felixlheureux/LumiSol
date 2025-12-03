import geopandas as gpd

# 1. Read the GeoJSON
gdf = gpd.read_file("data/montreal/uniteevaluationfonciere.geojson")

# 2. Save as GeoPackage
# 'layer' name is usually the file name without extension
gdf.to_file("data/montreal/lot_cadastre.gpkg", driver="GPKG", layer="lot_cadastre")

print("Conversion complete!")