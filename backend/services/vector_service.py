import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import Transformer
from scripts.pipeline.config import CONFIG

class VectorService:
    def __init__(self):
        print("🏛️  Initializing Government Vector Database...")
        self.vector_path = CONFIG["INPUT_VECTOR_PATH"]
        
        # Load the Spatial Index only (Lazy Load) if possible, 
        # or load the dataframe if it fits in RAM (standard for MVP).
        try:
            self.gdf = gpd.read_file(self.vector_path)
            # Ensure we are in Metric CRS (MTM8) for fast distance calcs
            if self.gdf.crs.to_string() != "EPSG:32188":
                self.gdf = self.gdf.to_crs("EPSG:32188")
            
            # Create a dedicated Spatial Index (Sindex) for millisecond lookups
            self.sindex = self.gdf.sindex
            print(f"✅ Loaded {len(self.gdf)} government building footprints.")
            
            # Transformer for Lat/Lon inputs
            self.to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32188", always_xy=True)
            self.to_web = Transformer.from_crs("EPSG:32188", "EPSG:4326", always_xy=True)
            
        except Exception as e:
            print(f"❌ Failed to load vectors: {e}")
            self.gdf = None

    def get_building_at_location(self, lat, lon):
        """
        Finds the building polygon containing the lat/lon.
        Returns: shapely.geometry.Polygon (in meters) or None
        """
        if self.gdf is None: return None

        # 1. Project Input -> Meters
        mx, my = self.to_metric.transform(lon, lat)
        point = Point(mx, my)
        
        # 2. Fast Spatial Query
        # intersects() is fast with the spatial index
        possible_matches_index = list(self.sindex.intersection(point.bounds))
        possible_matches = self.gdf.iloc[possible_matches_index]
        
        # 3. Exact Point-in-Polygon Check
        precise_match = possible_matches[possible_matches.contains(point)]
        
        if not precise_match.empty:
            return precise_match.geometry.iloc[0]
        
        # Fallback: If user clicked 'near' a building (within 10m), grab closest
        # This improves UX for fat-finger clicks on mobile
        buffer_region = point.buffer(10) # 10 meters radius
        near_matches_idx = list(self.sindex.intersection(buffer_region.bounds))
        near_matches = self.gdf.iloc[near_matches_idx]
        
        if not near_matches.empty:
            # Return closest
            # Calculate distance to point for all near matches
            distances = near_matches.distance(point)
            closest_idx = distances.idxmin()
            return self.gdf.loc[closest_idx].geometry
            
        return None
