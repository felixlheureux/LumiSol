import geopandas as gpd
import pyogrio

FILEPATH = "quebec_lots.gpkg"

def check_file():
    print(f"🕵️‍♂️ Inspecting: {FILEPATH}")
    
    try:
        # 1. List all layers in the file
        layers = pyogrio.list_layers(FILEPATH)
        # pyogrio returns a list of [name, type]
        layer_names = [l[0] for l in layers]
        print(f"📚 Layers found: {layer_names}")
        
        for layer_name in layer_names:
            print(f"\n--- Layer: {layer_name} ---")
            
            # 2. Read just 1 row to check geometry type
            gdf = gpd.read_file(FILEPATH, layer=layer_name, rows=1)
            geom_type = gdf.geometry.type.iloc[0] if not gdf.empty else "Empty"
            
            print(f"   📐 Geometry Type: {geom_type.upper()}")
            
            if "POLYGON" in geom_type.upper():
                print("   ✅ This is the layer you want!")
            elif "POINT" in geom_type.upper():
                print("   ❌ This is just dots (Centroids).")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    check_file()
