import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from app.core.config import LIDAR_FILE

def test_upscaling():
    # Load a small chunk of LiDAR
    with rasterio.open(str(LIDAR_FILE)) as src:
        # Center crop
        center_x, center_y = src.width // 2, src.height // 2
        window = rasterio.windows.Window(center_x - 50, center_y - 50, 100, 100)
        data = src.read(1, window=window)
        
    # Normalize for visualization
    data = np.nan_to_num(data, nan=np.min(data))
    
    # 1. Scipy Cubic (Current)
    cubic = zoom(data, 4, order=3)
    
    # 2. OpenCV Lanczos
    h, w = data.shape
    lanczos = cv2.resize(data, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    
    # 3. OpenCV Cubic
    cv_cubic = cv2.resize(data, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    # 4. Lanczos + Bilateral Filter (Edge Preserving Smoothing)
    # Bilateral requires float32
    lanczos_32 = lanczos.astype(np.float32)
    # sigmaColor: large to smooth height differences? No, small to preserve steps.
    # sigmaSpace: spatial extent.
    bilateral = cv2.bilateralFilter(lanczos_32, d=9, sigmaColor=1.0, sigmaSpace=75)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(cubic, cmap='terrain')
    axes[0].set_title("Scipy Cubic (Order 3)")
    
    axes[1].imshow(cv_cubic, cmap='terrain')
    axes[1].set_title("OpenCV Cubic")
    
    axes[2].imshow(lanczos, cmap='terrain')
    axes[2].set_title("OpenCV Lanczos4")
    
    axes[3].imshow(bilateral, cmap='terrain')
    axes[3].set_title("Lanczos + Bilateral")
    
    plt.savefig("upscaling_comparison.png")
    print("Saved upscaling_comparison.png")

if __name__ == "__main__":
    test_upscaling()
