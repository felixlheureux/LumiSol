import base64
import cv2
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import rasterio.features
from rasterio.features import rasterize

# Local Pipeline Modules
from scripts.pipeline.config import CONFIG
from scripts.pipeline.post_processor import PostProcessor
from services.geo_engine import GeoEngine
from services.solar_engine import SolarEngine
from services.vector_service import VectorService 
from services.alignment_engine import AlignmentEngine

app = FastAPI()

# 1. CORS (Allow Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global State Cache
state = {}

@app.on_event("startup")
async def startup_event():
    print("🚀 Booting LumiSol Backend...")
    try:
        # Load Engines
        state["geo"] = GeoEngine()
        state["solar"] = SolarEngine(CONFIG)
        state["vectors"] = VectorService()
        state["aligner"] = AlignmentEngine() # <--- NEW: Init Aligner
        
        # 💡 NOTE: We COMMENT OUT the AI loading for now to save startup time/RAM
        # state["post"] = PostProcessor(CONFIG)
        # model_path = CONFIG["MODEL_OUTPUT_DIR"]
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        # state["proc"] = Mask2FormerImageProcessor.from_pretrained(model_path, do_normalize=False, do_resize=False, do_rescale=True)
        # state["model"] = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
        # state["model"].eval()
        print("✅ MVP System Online (Gov Mode + Aligner).")
    except Exception as e:
        print(f"❌ Startup Error: {e}")

class AnalysisRequest(BaseModel):
    lat: float
    lon: float

@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    # 1. Gov Data Lookup (The "Truth")
    gov_polygon = state["vectors"].get_building_at_location(req.lat, req.lon)
    
    if not gov_polygon:
        raise HTTPException(404, "No building found in government registry at this location.")

    try:
        # 2. Fetch LiDAR Data (The "Physics")
        # We still need the raster to know the height/slope
        tensor, meta, raw_height = state["geo"].get_patch(req.lat, req.lon)
        
        # 3. Create Physics Mask
        # We burn the Government Polygon into the LiDAR grid
        mask_shape = raw_height.shape
        roof_mask = rasterize(
            [gov_polygon], 
            out_shape=mask_shape, 
            transform=meta["transform"]
        ).astype(bool)
        
        # 4. Extract Real-World Physics from Raster
        # Calculate Slope/Aspect on the raw height map
        dy, dx = np.gradient(raw_height, 0.2) # 0.2m pixel
        slope_grid = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect_grid = np.degrees(np.arctan2(-dx, dy)) % 360
        
        # Sample pixels ONLY inside the government footprint
        if np.any(roof_mask):
            avg_slope = np.mean(slope_grid[roof_mask])
            
            # Vector Average for Aspect
            aspect_rad = np.radians(aspect_grid[roof_mask])
            avg_aspect = np.degrees(np.arctan2(
                np.mean(np.sin(aspect_rad)), 
                np.mean(np.cos(aspect_rad))
            )) % 360
        else:
            # Fallback if polygon is off-tile (rare)
            avg_slope = 0
            avg_aspect = 180

        # 5. Run Solar Simulation (365 Days)
        # Use the GOV Area + LIDAR Slope/Aspect
        graph_data, total_kwh = state["solar"].simulate_year(
            req.lat, req.lon, avg_slope, avg_aspect, gov_polygon.area
        )

        # 6. Visualization (Heatmap)
        # We visualize the LiDAR height map, but outline the Gov Polygon in Blue
        vis_img = cv2.applyColorMap(tensor[:,:,0], cv2.COLORMAP_JET)
        
        # Helper: World -> Pixel
        def world_to_px(x, y):
            row, col = ~meta["transform"] * (x, y)
            return int(row), int(col)

        # Draw Gov Polygon
        if not gov_polygon.is_empty:
            pts = np.array([world_to_px(x, y) for x, y in gov_polygon.exterior.coords], dtype=np.int32)
            # Note: rasterio (row, col) -> cv2 (x, y) might need swap depending on orientation
            # usually row=y, col=x. cv2 uses (x,y).
            pts = pts[:, [1, 0]] # Swap to (col, row) -> (x, y)
            cv2.polylines(vis_img, [pts], True, (255, 255, 0), 2) # Cyan/Blue for Gov Data
        
        _, buf = cv2.imencode('.png', vis_img)
        img_str = base64.b64encode(buf).decode('utf-8')

        # 7. Coordinate Conversion for Frontend
        transformer = state["geo"].to_web
        minx, miny = meta["bounds"][0]
        maxx, maxy = meta["bounds"][1]
        w, s = transformer.transform(minx, miny)
        e, n = transformer.transform(maxx, maxy)
        
        # --- NEW: Calculate & Apply Shift ---
        # Run SAM to find the "Visual Roof" vs "Gov Roof" offset
        try:
            # shift_x, shift_y are in the web projection units (Lat/Lon)
            dx, dy = state["aligner"].calculate_shift(req.lat, req.lon, gov_polygon)
            
            # Apply shift to the bounds we send to Frontend
            # This moves the overlay image to match the satellite background
            w += dx
            e += dx
            s += dy
            n += dy
            print(f"   ✨ Applied Visual Alignment: {dx:.6f}, {dy:.6f}")
        except Exception as e:
            print(f"   ⚠️ Alignment failed, using raw coordinates: {e}")

        # Return Result
        return {
            "heatmap": f"data:image/png;base64,{img_str}",
            "bounds": [[w, s], [e, n]],
            "solar_potential": f"{total_kwh:,} kWh/yr",
            "area": f"{int(gov_polygon.area)} m²",
            "graph_data": graph_data,
            "lot_polygon": [] # Optional: Pass lot boundary if you have it
        }

    except Exception as e:
        print(e)
        raise HTTPException(500, f"Analysis failed: {str(e)}")
