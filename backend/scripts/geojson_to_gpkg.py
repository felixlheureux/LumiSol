import geopandas as gpd
import os
import time

# CONFIG
# Input: The Montreal GeoJSON you downloaded (make sure it's the 165MB one)
INPUT_FILE = "uniteevaluationfoncieremontreal.geojson" 
# Output: The file your backend logic looks for
OUTPUT_FILE = "quebec_lots.gpkg"

def convert_geojson_to_gpkg():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find '{INPUT_FILE}'")
        print("   Please download the 'Extrait géographique' (GeoJSON) from Données Montréal.")
        return

    print(f"📖 Reading {INPUT_FILE}... (This may take 10-20 seconds)")
    start = time.time()
    
    # 1. Load GeoJSON
    gdf = gpd.read_file(INPUT_FILE)
    
    # Verify it is polygons
    if not gdf.empty:
        geom_type = gdf.geometry.type.iloc[0]
        print(f"   -> Loaded {len(gdf)} records.")
        print(f"   -> Geometry Type: {geom_type}")
        
        if "POLYGON" not in geom_type.upper():
            print("❌ STOP: This file also contains Points! Download the correct file.")
            return
    else:
        print("❌ Error: The file is empty.")
        return

    # 2. Save as GeoPackage (GPKG)
    print(f"💾 Converting to {OUTPUT_FILE}...")
    # We use 'lots' as the layer name so the engine knows where to look
    gdf.to_file(OUTPUT_FILE, driver="GPKG", layer="lots")
    
    elapsed = time.time() - start
    print(f"✅ Success! Created database in {elapsed:.1f} seconds.")
    print("   You can now restart your backend server.")

if __name__ == "__main__":
    convert_geojson_to_gpkg()