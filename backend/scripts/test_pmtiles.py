import geopandas as gpd
import pyogrio

def test_pmtiles():
    path = "quebec_lots.pmtiles"
    print(f"Testing read of {path}...")
    try:
        # Try reading info first
        layers = pyogrio.list_layers(path)
        print("Layers:", layers)
        
        from pyproj import Transformer
        
        # Target: 45.551126, -73.544592
        lat, lon = 45.551126, -73.544592
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        print(f"Target in 3857: {x}, {y}")
        
        # BBox +/- 50 meters
        bbox = (x - 50, y - 50, x + 50, y + 50)
        
        print(f"Querying bbox: {bbox}")
        gdf = gpd.read_file(path, bbox=bbox)
        print(f"Read {len(gdf)} features.")
        if not gdf.empty:
            print("Geometry Type:", gdf.geometry.type.unique())
            print(gdf.head())
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_pmtiles()
