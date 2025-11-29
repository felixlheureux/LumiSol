import geopandas as gpd
from shapely.geometry import Point, box
import os
from pyproj import Transformer
from app.core.config import CADASTRE_FILE

class LotManager:
    def __init__(self, file_path=None):
        """
        Initializes the LotManager.
        For GeoJSON, loads the file into memory.
        For PMTiles or GPKG, just stores the path (lazy loading).
        """
        if file_path is None:
            self.file_path = str(CADASTRE_FILE)
        else:
            self.file_path = str(file_path)
            
        self.is_pmtiles = self.file_path.endswith(".pmtiles")
        self.is_gpkg = self.file_path.endswith(".gpkg")
        self.gdf = None
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        # Only load GeoJSON into memory
        if self.file_path.endswith(".geojson"):
            print(f"Loading Lot Data from {self.file_path}...")
            self.gdf = gpd.read_file(self.file_path)
            if self.gdf.crs is None:
                self.gdf.set_crs(epsg=4326, inplace=True)
            else:
                self.gdf = self.gdf.to_crs(epsg=4326)
            print(f"✅ Loaded {len(self.gdf)} lots.")
        else:
            print(f"Initialized LotManager with {self.file_path} (Lazy Loading)")

    def get_lot_at_point(self, lat, lon, target_crs="EPSG:2950"):
        """
        Finds the lot polygon containing the point (lat, lon).
        Returns the polygon coordinates projected to target_crs.
        """
        if self.file_path.endswith(".geojson"):
            # GeoJSON Strategy (Exact Polygon - full in-memory GDF)
            p = Point(lon, lat)
            # Use spatial index if available, else linear scan
            if self.gdf is not None:
                possible_matches_index = list(self.gdf.sindex.query(p, predicate='contains'))
                possible_matches = self.gdf.iloc[possible_matches_index]
            else:
                return None
            
            if not possible_matches.empty:
                lot = possible_matches.iloc[0]
                # Reproject to target CRS
                lot_geom = gpd.GeoSeries([lot.geometry], crs=self.gdf.crs)
                lot_geom_projected = lot_geom.to_crs(target_crs)
                return list(lot_geom_projected[0].exterior.coords)
            return None
            
        elif self.is_gpkg:
            # GeoPackage Strategy (Real Polygons - queried on demand)
            # CRS is likely EPSG:4269 (NAD83 Lat/Lon)
            
            # Create a buffer around the point (e.g. 0.001 degrees ~ 100m)
            delta = 0.001
            bbox_4269 = (lon - delta, lat - delta, lon + delta, lat + delta)
            
            try:
                # Read features in bbox. 
                # The user created the GPKG with layer="lots".
                # We try 'lots' first, then fallback to default (first layer).
                try:
                    possible_lots = gpd.read_file(self.file_path, bbox=bbox_4269, layer='lots')
                except ValueError:
                    # Layer 'lots' not found, try default
                    possible_lots = gpd.read_file(self.file_path, bbox=bbox_4269)
            except Exception as e:
                print(f"Error reading GPKG with bbox: {e}")
                return None
                
            if len(possible_lots) == 0:
                print("No lots found in bbox.")
                return None
                
            # Find the polygon that actually contains the point
            point = Point(lon, lat)
            
            # Ensure CRS match
            if possible_lots.crs and possible_lots.crs != "EPSG:4326" and possible_lots.crs != "EPSG:4269":
                 # If file is projected, we must project the point
                 transformer = Transformer.from_crs("EPSG:4326", possible_lots.crs, always_xy=True)
                 tx, ty = transformer.transform(lon, lat)
                 point = Point(tx, ty)
            
            # Filter for containment
            containing_lot = possible_lots[possible_lots.contains(point)]
            
            if len(containing_lot) > 0:
                lot = containing_lot.iloc[0]
            else:
                # Fallback: Find closest lot (e.g. if point is on street)
                print("Point not inside any lot. Finding closest lot...")
                possible_lots['distance'] = possible_lots.distance(point)
                lot = possible_lots.loc[possible_lots['distance'].idxmin()]
                
            # Reproject to target CRS
            lot_geom = gpd.GeoSeries([lot.geometry], crs=possible_lots.crs)
            lot_geom_projected = lot_geom.to_crs(target_crs)
            
            final_geom = lot_geom_projected.iloc[0]
            
            # Handle MultiPolygon
            if final_geom.geom_type == 'MultiPolygon':
                # Take largest polygon
                final_geom = max(final_geom.geoms, key=lambda a: a.area)
            
            # Check if Point (shouldn't happen with 'lots' layer but safe to keep)
            if 'Point' in final_geom.geom_type:
                print("⚠️ Warning: Found Point geometry in GPKG. Buffering...")
                lot_poly = final_geom.buffer(30, cap_style=3)
                return list(lot_poly.exterior.coords)
            else:
                return list(final_geom.exterior.coords)
            
        elif self.is_pmtiles:
            # PMTiles Strategy (Points -> Synthetic Polygon)
            # 1. Convert Target to EPSG:3857 (Web Mercator)
            transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer_to_3857.transform(lon, lat)
            
            # 2. Query BBox (+/- 20m)
            search_radius = 20
            bbox = (x - search_radius, y - search_radius, x + search_radius, y + search_radius)
            
            try:
                gdf = gpd.read_file(self.file_path, bbox=bbox)
            except Exception as e:
                print(f"Error reading PMTiles: {e}")
                return None
                
            if gdf.empty:
                return None
                
            # 3. Find Closest Point
            # We calculate distance in 3857 (meters)
            target_point = Point(x, y)
            gdf['distance'] = gdf.geometry.distance(target_point)
            closest_lot = gdf.loc[gdf['distance'].idxmin()]
            
            # 4. Create Synthetic Polygon (Buffer)
            # Since we only have points, we simulate a lot boundary
            # Let's assume a 30m radius (approx 2800m^2 lot) to catch the house
            # Or a square box
            lot_geom_3857 = closest_lot.geometry.buffer(30, cap_style=3) # cap_style=3 is Square
            
            # 5. Reproject to Target CRS
            lot_series = gpd.GeoSeries([lot_geom_3857], crs="EPSG:3857")
            lot_projected = lot_series.to_crs(target_crs)
            
            return list(lot_projected[0].exterior.coords)
        else:
            print(f"No strategy implemented for file type: {self.file_path}")
            return None
