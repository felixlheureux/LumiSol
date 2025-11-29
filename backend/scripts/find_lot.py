import geopandas as gpd
from shapely.geometry import Point

def find_valid_lot():
    print("Loading GeoJSON...")
    gdf = gpd.read_file("uniteevaluationfonciere_montreal.geojson")
    
    # Get the first polygon
    first_lot = gdf.iloc[0]
    centroid = first_lot.geometry.centroid
    
    print(f"Found valid lot ID: {first_lot.get('ID_UEV', 'Unknown')}")
    print(f"Centroid (Lat, Lon): {centroid.y}, {centroid.x}")
    
    # Also check bounds to see if our test point is way off
    print(f"Total Bounds: {gdf.total_bounds}")

if __name__ == "__main__":
    find_valid_lot()
