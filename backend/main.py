import io
import base64
import cv2
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# Local Pipeline Modules
from scripts.pipeline.config import CONFIG
from scripts.pipeline.post_processor import PostProcessor
from services.geo_engine import GeoEngine

app = FastAPI()

# 1. CORS Setup (Allow Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only. Restrict in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global State
state = {
    "model": None,
    "processor": None,
    "geo_engine": None,
    "post_processor": None
}

@app.on_event("startup")
async def startup_event():
    print("🚀 Booting LumiSol Backend...")
    
    # Init GeoEngine (Loads Index)
    try:
        state["geo_engine"] = GeoEngine()
        state["post_processor"] = PostProcessor(CONFIG)
    except Exception as e:
        print(f"❌ Core services failed: {e}")

    # Init AI Model
    model_path = CONFIG["MODEL_OUTPUT_DIR"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        state["processor"] = Mask2FormerImageProcessor.from_pretrained(
            model_path, do_normalize=False, do_resize=False, do_rescale=True
        )
        state["model"] = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
        state["model"].eval()
        print("✅ AI Model Ready.")
    except:
        print(f"⚠️  Model not found at {model_path}. Did you run 'lumisol.py train'?")

class AnalysisRequest(BaseModel):
    lat: float
    lon: float

@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    if not state["model"]:
        raise HTTPException(503, "AI Engine not online. Train the model first.")

    try:
        # A. Fetch Data (On-Demand)
        tensor, meta = state["geo_engine"].get_patch(req.lat, req.lon)
        
        # B. Run Inference
        device = state["model"].device
        image_rgb = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB) # Model expects RGB input
        
        inputs = state["processor"](images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = state["model"](**inputs)
        
        # C. Decode Result
        target_size = [(tensor.shape[0], tensor.shape[1])]
        results = state["processor"].post_process_instance_segmentation(
            outputs, target_sizes=target_size
        )[0]
        
        # Extract the binary mask (Collapse all instances to 1 layer for now)
        pred_map = results["segmentation"].cpu().numpy()
        binary_mask = (pred_map > -1).astype(np.uint8)
        
        # D. Post-Process (Dilate & Regularize)
        # Note: We pass None as CRS here to get Pixel Coordinates for the frontend display
        # Ideally, you'd do this twice: once in Pixels (for Heatmap), once in Meters (for Area calc)
        refined_polys_px = state["post_processor"].process_single_mask(binary_mask)
        
        # E. Calculate Metrics
        total_area_px = sum([p.area for p in refined_polys_px])
        pixel_res_sq = 0.2 * 0.2 # 0.04 m2 per pixel
        real_area = total_area_px * pixel_res_sq
        
        # F. Generate Heatmap Image (Base64)
        # We visualize the Input Height (Red Channel) + Green Contours
        heatmap_img = cv2.applyColorMap(tensor[:,:,0], cv2.COLORMAP_JET)
        
        # Draw Polygons
        vis_mask = state["post_processor"].polygons_to_image(refined_polys_px, tensor.shape[:2])
        # Overlay: Green tint where building is
        heatmap_img[vis_mask > 0] = heatmap_img[vis_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        _, buffer = cv2.imencode('.png', heatmap_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # G. Calculate GeoBounds for the Frontend Overlay
        # Convert Metric Bounds -> Lat/Lon
        minx, miny = meta["bounds"][0]
        maxx, maxy = meta["bounds"][1]
        transformer = state["geo_engine"].to_web
        
        w, s = transformer.transform(minx, miny)
        e, n = transformer.transform(maxx, maxy)

        return {
            "heatmap": f"data:image/png;base64,{img_str}",
            "bounds": [[w, s], [e, n]],
            "solar_potential": f"{int(real_area * 150)} kWh/yr", # Mock physics (150kWh/m2)
            "area": f"{int(real_area)} m²",
            "lot_polygon": [] # Can calculate this if vector_filter passed lot geometry
        }

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        print(e)
        raise HTTPException(500, "Internal Engine Error")
