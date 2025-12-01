import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import time
import json
import random
import os

class VectorFilter:
    def __init__(self, config):
        self.config = config
        self.montreal_bbox = box(-73.9, 45.4, -73.5, 45.7)

    def run(self):
        print(f"🚀 Starting Smart Vector Filtering...")
        
        # Ensure output directories exist
        Path(self.config['OUTPUT_VECTOR_PATH']).parent.mkdir(parents=True, exist_ok=True)
        vectors_cache_dir = os.path.join(self.config['CACHE_DIR'], "vectors")
        Path(vectors_cache_dir).mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # 1. Load Vectors with BBOX Filter (Montreal Region)
        print("1. Loading Montreal Vectors...")
        try:
            # Peek CRS to project BBOX
            meta = gpd.read_file(self.config['INPUT_VECTOR_PATH'], rows=1)
            source_crs = meta.crs
            
            bbox_gdf = gpd.GeoDataFrame({'geometry': [self.montreal_bbox]}, crs="EPSG:4326")
            if source_crs and source_crs != "EPSG:4326":
                bbox_gdf = bbox_gdf.to_crs(source_crs)
            
            target_bbox = bbox_gdf.geometry.iloc[0].bounds
            
            # Load all buildings in Montreal
            buildings = gpd.read_file(self.config['INPUT_VECTOR_PATH'], bbox=target_bbox)
            print(f"   ✅ Loaded {len(buildings)} buildings in Montreal region.")
            
            if buildings.empty:
                print("   ⚠️ Warning: No buildings found in this region.")
                return

            # 2. Load LiDAR Index
            print("2. Loading LiDAR Index...")
            with open(self.config['LIDAR_INDEX_PATH']) as f: data = json.load(f)
            crs = data.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
            lidar_index = gpd.GeoDataFrame.from_features(data['features'], crs=crs)

            # Project to match vectors
            if lidar_index.crs != buildings.crs:
                print("   Reprojecting LiDAR index...")
                lidar_index = lidar_index.to_crs(buildings.crs)

            # 3. Spatial Join & Sampling
            print("3. Selecting Tiles...")
            lidar_with_counts = gpd.sjoin(lidar_index, buildings, how="inner", predicate="intersects")
            tile_counts = lidar_with_counts.groupby(lidar_with_counts.index).size().reset_index(name='count')
            
            valid_indices = tile_counts['index'].tolist()
            print(f"   Found {len(valid_indices)} valid tiles with buildings.")

            # Stratified Sampling
            high = tile_counts[tile_counts['count'] > 2000]['index'].tolist()
            med = tile_counts[(tile_counts['count'] >= 200) & (tile_counts['count'] <= 2000)]['index'].tolist()
            low = tile_counts[tile_counts['count'] < 200]['index'].tolist()
            
            print(f"   High: {len(high)} | Med: {len(med)} | Low: {len(low)}")
            
            random.seed(self.config['RANDOM_SEED'])
            selected_indices = set()
            
            remaining_needed = self.config['MAX_TILES']
            target_per_cat = max(1, remaining_needed // 3)
            
            def pick_new(pool, n):
                available = list(set(pool) - selected_indices)
                return random.sample(available, min(len(available), n))

            selected_indices.update(pick_new(high, target_per_cat))
            selected_indices.update(pick_new(med, target_per_cat))
            selected_indices.update(pick_new(low, target_per_cat))
            
            still_needed = self.config['MAX_TILES'] - len(selected_indices)
            if still_needed > 0:
                all_avail = set(valid_indices) - selected_indices
                selected_indices.update(random.sample(list(all_avail), min(len(all_avail), still_needed)))
                
            print(f"   🎯 Selected {len(selected_indices)} tiles.")

            # 4. Save Selected Tiles Metadata
            selected_tiles_data = []
            selected_lidar_rows = lidar_index.loc[list(selected_indices)]
            
            for idx, lidar_row in selected_lidar_rows.iterrows():
                lidar_url = lidar_row.get('MHC') or lidar_row.get('MNS') or lidar_row.get('MNT')
                if not lidar_url: continue
                
                selected_tiles_data.append({
                    "id": idx,
                    "filename": f"lidar_{idx}.tif",
                    "url": lidar_url
                })
                
            with open(self.config['OUTPUT_TILES_PATH'], 'w') as f:
                json.dump(selected_tiles_data, f, indent=2)
            print(f"   ✅ Saved tile list to {self.config['OUTPUT_TILES_PATH']}")

            # 5. Filter Vectors to Selected Tiles & Save Chunks
            print("5. Filtering Vectors & Saving Chunks...")
            
            # We still save the combined one for reference/backup, but the main usage will be chunks
            selected_tiles_geom = selected_lidar_rows.unary_union
            
            possible_matches = list(buildings.sindex.intersection(selected_tiles_geom.bounds))
            candidate_buildings = buildings.iloc[possible_matches]
            final_buildings = candidate_buildings[candidate_buildings.intersects(selected_tiles_geom)]
            
            print(f"   ✅ Kept {len(final_buildings)} buildings (from {len(buildings)}).")

            # Save Combined (Optional, but good for visualization)
            print("   Saving combined GeoPackage...")
            final_buildings.to_file(self.config['OUTPUT_VECTOR_PATH'], driver="GPKG")
            
            # Save Individual Chunks
            print("   Saving per-tile vector chunks...")
            for idx, lidar_row in selected_lidar_rows.iterrows():
                tile_id = idx
                chunk_path = os.path.join(vectors_cache_dir, f"vectors_{tile_id}.gpkg")
                
                if os.path.exists(chunk_path):
                    # print(f"      Skipping existing chunk: {chunk_path}")
                    continue
                
                # Filter for this specific tile
                # Use the tile geometry from the index
                tile_geom = lidar_row.geometry
                
                # Spatial query on the ALREADY FILTERED final_buildings
                # This is fast
                chunk_buildings = final_buildings[final_buildings.intersects(tile_geom)]
                
                if not chunk_buildings.empty:
                    chunk_buildings.to_file(chunk_path, driver="GPKG")
            
            print(f"   ✅ Saved chunks to {vectors_cache_dir}")
            print(f"   ⏱️ Time taken: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
