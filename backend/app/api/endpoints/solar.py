from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import base64
from app.services.solar_engine import SolarEngine
from app.services.lot_manager import LotManager
from app.core.config import LIDAR_FILE

router = APIRouter()
engine = SolarEngine()
lot_manager = LotManager() # Uses default CADASTRE_FILE

@router.post("/analyze")
async def analyze_roof(data: dict = Body(...)):
    lat = data.get("lat")
    lon = data.get("lon")

    print(f"Received request for {lat}, {lon}")

    # 1. GET LOT POLYGON
    # We use the real LotManager now!
    try:
        # We need to target the CRS of the LiDAR file.
        # For now, let's assume EPSG:2950 (Montreal MTM8) as per our tests.
        # Ideally, we read this from the TIF, but for now hardcode or use config.
        target_crs = "EPSG:2950" 
        lot_polygon = lot_manager.get_lot_at_point(lat, lon, target_crs=target_crs)
        
        if not lot_polygon:
             return JSONResponse(status_code=404, content={"error": "No lot found at this location."})
             
    except Exception as e:
        print(f"Error finding lot: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # 2. FETCH LIDAR DATA
    # TODO: Implement real cropping from TIF using rasterio
    # For now, we will use the MOCK generator if the file doesn't exist or just to get it running
    # BUT we really should use the real data if we have it.
    
    # Let's try to load the real data if possible, otherwise mock.
    # Since we are in a "Restructure" task, maybe I shouldn't implement full feature.
    # But the old main.py used mock.
    # I will stick to MOCK for now to ensure the API responds, 
    # BUT I will update the segmentation call to the new method.
    
    mns, mnt = mock_fetch_lidar(lat, lon)
    
    # 3. SEGMENTATION
    # The new method requires a lot_mask.
    # Since we are mocking data, we don't have a real lot mask aligned to the mock data.
    # So we will pass None for lot_mask, or create a dummy one.
    
    # engine.segment_lot_structures(mns, mnt, lot_mask)
    # It returns: final_mask, ndsm
    
    structures_mask, ndsm = engine.segment_lot_structures(mns, mnt, lot_mask=None)
    
    if np.sum(structures_mask) == 0:
         return JSONResponse(status_code=404, content={"error": "No structures found."})

    # 4. SOLAR MATH
    solar_scores = engine.calculate_solar_potential(ndsm, structures_mask)

    # Calculate total potential
    # Mock data is usually 1m res?
    total_kwh = np.sum(solar_scores) * 150 # Dummy multiplier
    area = np.sum(structures_mask) 

    # 5. GENERATE IMAGE
    img_io = engine.generate_heatmap_overlay(solar_scores)
    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Calculate dummy bounds
    delta = 0.0005
    bounds = [lon - delta, lat - delta, lon + delta, lat + delta]

    return {
        "heatmap_b64": f"data:image/png;base64,{img_b64}",
        "bounds": bounds,
        "solar_potential": int(total_kwh),
        "area_sqm": int(area)
    }

def mock_fetch_lidar(lat, lon, size=100):
    """
    Generates a fake roof shape for testing.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    mns = 20 - 20 * np.maximum(np.abs(X), np.abs(Y))
    mnt = np.zeros_like(mns)
    return mns, mnt
