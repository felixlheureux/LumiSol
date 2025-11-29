import numpy as np
from solar_engine import SolarEngine
from main import mock_fetch_lidar
import matplotlib.pyplot as plt

def test_segmentation():
    print("Testing Roof Segmentation...")
    
    # 1. Generate Mock Data (Pyramid)
    # Center is at 50, 50. It has 4 faces.
    mns, mnt = mock_fetch_lidar(45.0, -73.0, size=100)
    
    engine = SolarEngine()
    
    # 2. Test Smart Snap
    # Click slightly off-center (e.g., 52, 52) which should be on one face
    click_x, click_y = 52, 52
    print(f"Clicking at: {click_x}, {click_y}")
    
    seed_x, seed_y = engine.find_smart_seed(mns, mnt, click_x, click_y)
    print(f"Smart Seed found: {seed_x}, {seed_y}")
    
    # Verify seed is reasonable (should be close to click but maybe optimized)
    # In a perfect pyramid, the highest point is 50,50. 
    # But smart seed balances Height vs Roughness vs Distance.
    # The peak might be 'rough' if it's a sharp point, but here it's smooth planes.
    
    # 3. Test Region Grow
    # We expect it to grab ONE face of the pyramid, not the whole thing.
    # Returns: High-Res Mask, High-Res MNS Crop, Barrier Map
    facet_mask, mns_high_res, _ = engine.segment_roofs(mns, mnt, click_x, click_y)
    
    if facet_mask is None:
        print(f"Segmentation Failed: {mns_high_res}")
        return

    print("Segmentation Successful!")
    print(f"Mask Area: {np.sum(facet_mask)} pixels")
    
    # Verify it didn't grab the whole pyramid
    # Note: mns_high_res is 4x larger (16x pixels)
    total_pyramid_area = np.sum(mns_high_res > 0)
    print(f"Total Pyramid Area (High-Res): {total_pyramid_area}")
    
    if np.sum(facet_mask) < total_pyramid_area * 0.4:
        print("PASS: Segmented a subset (likely one face).")
    else:
        print("FAIL: Segmented too much (likely the whole pyramid).")

if __name__ == "__main__":
    test_segmentation()
