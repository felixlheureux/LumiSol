from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import base64
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import minimum_filter, gaussian_filter
from pyproj import Transformer
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
    try:
        # Target CRS: EPSG:2950 (Montreal MTM8) - Hardcoded for this dataset
        target_crs = "EPSG:2950" 
        lot_polygon_geom = lot_manager.get_lot_at_point(lat, lon, target_crs=target_crs)
        
        if not lot_polygon_geom:
             return JSONResponse(status_code=404, content={"error": "No lot found at this location."})

        # Convert to list of points for frontend (WGS84)
        if lot_polygon_geom.geom_type == 'Polygon':
            exterior_coords = list(lot_polygon_geom.exterior.coords)
        elif lot_polygon_geom.geom_type == 'MultiPolygon':
            largest = max(lot_polygon_geom.geoms, key=lambda p: p.area)
            exterior_coords = list(largest.exterior.coords)
        else:
            exterior_coords = []
            
        # Transform to WGS84 for frontend display
        # Input is EPSG:2950, Output is EPSG:4326
        project_to_wgs84 = Transformer.from_crs("EPSG:2950", "EPSG:4326", always_xy=True).transform
        wgs84_coords = [project_to_wgs84(x, y) for x, y in exterior_coords]
             
    except Exception as e:
        print(f"Error finding lot: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # 2. Fetch Real LiDAR Data
    try:
        with rasterio.open(str(LIDAR_FILE)) as src:
            # Define a window around the lot
            # We use the lot bounds + padding
            minx, miny, maxx, maxy = lot_polygon_geom.bounds
            padding = 10 # meters
            
            # Ensure window coordinates are within src bounds
            window = from_bounds(minx - padding, miny - padding, maxx + padding, maxy + padding, src.transform)
            window = window.round_offsets().round_lengths() # Ensure integer offsets/lengths
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height)) # Clip to dataset bounds
            
            # Read data
            mns = src.read(1, window=window)
            
            # Handle NoData
            mns = np.nan_to_num(mns, nan=np.nanmin(mns))
            
            # Synthetic MNT (Ground Estimate) - Since we don't have a real DTM file yet
            # We use a minimum filter to find the "ground"
            ground_estimate = minimum_filter(mns, size=20)
            mnt = gaussian_filter(ground_estimate, sigma=2)
            
            transform = src.window_transform(window)
            
            # Calculate Bounds for Frontend Overlay
            # We need the bounds of the window we read
            win_bounds = src.window_bounds(window) # (left, bottom, right, top) in EPSG:2950
            
            # Reproject bounds to WGS84
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            sw = transformer.transform(win_bounds[0], win_bounds[1]) # minx, miny
            ne = transformer.transform(win_bounds[2], win_bounds[3]) # maxx, maxy
            
            # MapLibre Bounds: [[west, south], [east, north]]
            # sw is (lon, lat), ne is (lon, lat)
            bounds = [[sw[0], sw[1]], [ne[0], ne[1]]]

    except Exception as e:
        print(f"Error reading LiDAR: {e}")
        raise HTTPException(status_code=500, detail=f"LiDAR error: {e}")

    # 3. PROCESS GEOMETRY & SEGMENTATION (Scientific Pipeline)
    # We get the Mask AND the Vector Field (nx, ny, nz)
    # Note: process_geometry is no longer needed as a separate step, 
    # segment_solar_facets handles the physics internally.
    
    # Create Lot Mask (if lot exists)
    # We need to scale the polygon to the new high-res grid
    # The engine upscales by 2x, so we need a transform that matches the upscaled data.
    new_transform = transform * transform.scale(0.5, 0.5) 
    lot_mask = engine.create_lot_mask(lot_polygon_geom, (mns.shape[0]*2, mns.shape[1]*2), new_transform)

    # 3. Segment Roofs
    # Now returns 6 values: mask, nx, ny, nz, ndsm, hillshade
    # Pass transform for AI Segmentation
    structures_mask, nx, ny, nz, _, _ = engine.segment_solar_facets(mns, mnt, lot_mask=lot_mask, transform=new_transform)

    if np.sum(structures_mask) == 0:
         return JSONResponse(status_code=200, content={
            "address": "Detected Lot",
            "solar_potential": "0 kWh/yr",
            "area": "0 m²",
            "heatmap": "",
            "bounds": bounds,
            "lot_polygon": wgs84_coords,
            "message": "No structures suitable for solar panels found."
         })

    # 4. CALCULATE ENERGY
    solar_scores = engine.calculate_irradiance(nx, ny, nz, structures_mask)

    # 5. STATISTICS
    # Calculate roof area (accounting for 2x super-resolution)
    pixel_area_m2 = (engine.raw_resolution / engine.upsample_factor) ** 2
    roof_area = np.sum(structures_mask) * pixel_area_m2
    
    # Average Score of the VIABLE area only (Don't average the north side!)
    avg_efficiency = np.mean(solar_scores[structures_mask])
    
    # Montreal Estimator: 1200 kWh/kWp/year is optimal. 
    # We scale this by our efficiency score.
    total_kwh = roof_area * avg_efficiency * 150 # 150 is a rough W/m2 panel yield factor

    # 8. RENDER HEATMAP
    img_io = engine.generate_heatmap(solar_scores)
    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    return {
        "address": "Detected Lot",
        "solar_potential": f"{total_kwh:.0f} kWh/yr",
        "area": f"{roof_area:.0f} m²",
        "heatmap": f"data:image/png;base64,{img_b64}",
        "bounds": bounds,
        "lot_polygon": wgs84_coords
    }
