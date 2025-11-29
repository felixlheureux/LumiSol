import numpy as np
import cv2
import scipy.ndimage
from scipy.ndimage import zoom, label, generate_binary_structure, gaussian_filter, binary_opening, binary_closing, binary_fill_holes
from matplotlib import cm
from io import BytesIO
import matplotlib.pyplot as plt
from rasterio.mask import mask as rio_mask
from rasterio import features

class SolarEngine:
    def __init__(self):
        self.raw_resolution = 1.0 
        # 2x is the "Sweet Spot". 4x adds compute lag with diminishing physics returns.
        self.upsample_factor = 2 

    def crop_lidar_to_lot(self, mns_src, mnt_src, lot_geometry):
        """
        Crops the LiDAR rasters using the Vector Lot Geometry.
        Returns masked arrays where outside pixels are NaN.
        """
        # Crop MNS (Surface)
        mns_image, transform = rio_mask(mns_src, [lot_geometry], crop=True)
        mns_data = mns_image[0].astype('float32')
        
        # Crop MNT (Terrain)
        mnt_image, _ = rio_mask(mnt_src, [lot_geometry], crop=True)
        mnt_data = mnt_image[0].astype('float32')

        # Mask NoData values
        # We use NaN for invalid data to prevent physics calculations on boundaries
        mns_data[mns_data == mns_src.nodata] = np.nan
        mnt_data[mnt_data == mnt_src.nodata] = np.nan

        return mns_data, mnt_data, transform

    def create_lot_mask(self, lot_polygon, shape, transform):
        """
        Creates a binary mask from the lot polygon.
        """
        mask = features.rasterize(
            [lot_polygon],
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8'
        )
        return mask.astype(bool)

    def calculate_derivatives(self, mns, cell_size):
        """
        Calculates Gradient (Slope) and Aspect (Direction) using vector calculus.
        Applies Gaussian Smoothing first to prevent interpolation artifacts.
        """
        # 1. Smooth the surface to fix 'stair-step' aliasing from upsampling
        # Sigma=0.5 pixels approximates the Nyquist limit for the new grid
        mns_smooth = gaussian_filter(mns, sigma=0.5)

        # 2. Calculate Gradients
        dy, dx = np.gradient(mns_smooth, cell_size)
        
        # 3. Calculate Surface Normal Components
        # Normal Vector N = [-dx, -dy, 1]
        # We normalize it to unit length
        magnitude = np.sqrt(dx**2 + dy**2 + 1)
        nx = -dx / magnitude
        ny = -dy / magnitude
        nz = 1.0 / magnitude
        
        # 4. Calculate Slope (Zenith Angle)
        # 0 = Flat, PI/2 = Vertical Wall
        slope = np.arccos(nz)
        
        return nx, ny, nz, slope

    def _estimate_terrain(self, mns_grid, window_size_meters=30):
        """
        Creates a 'Rolling Ball' Digital Terrain Model (DTM).
        This ignores houses and finds the underlying ground, even on slopes.
        """
        # Convert meters to pixels (e.g. 30m / 0.5m = 60 pixels)
        pixels = int(window_size_meters / (self.raw_resolution / self.upsample_factor))
        
        # 1. Minimum Filter (Erosion): Finds the lowest pixel in the neighborhood
        rough_terrain = scipy.ndimage.minimum_filter(mns_grid, size=pixels)
        
        # 2. Gaussian Filter: Smooths the blocky erosion artifacts
        smooth_terrain = gaussian_filter(rough_terrain, sigma=pixels/4)
        
        return smooth_terrain

    def segment_solar_facets(self, mns, mnt, lot_mask):
        """
        PRODUCTION SEGMENTATION:
        1. Bilateral Filter (Sharpen Edges / Smooth Noise).
        2. Marker-Controlled Watershed (Topological Segmentation).
        3. Douglas-Peucker Regularization (Vector Simplification).
        """
        # --- 1. UPSAMPLING & PRE-PROCESSING ---
        mns_high = zoom(mns, self.upsample_factor, order=1)
        
        # [NEW] Bilateral Filter
        # This is the "Magic Sauce". It smooths texture (shingles) but PRESERVES edges.
        # d=5: Look at 5 pixel neighborhood
        # sigmaColor=0.5: Only mix pixels if height diff < 0.5m (Preserves walls)
        # sigmaSpace=75: Smooth broad flat areas
        mns_high = mns_high.astype(np.float32)
        mns_high = cv2.bilateralFilter(mns_high, d=5, sigmaColor=0.5, sigmaSpace=75)

        # Dynamic Terrain
        mnt_high = self._estimate_terrain(mns_high, window_size_meters=30)
        
        # Lot Mask
        if lot_mask is not None:
            if lot_mask.shape == mns_high.shape:
                lot_mask_high = lot_mask
            else:
                lot_mask_high = zoom(lot_mask, self.upsample_factor, order=0)
        else:
            lot_mask_high = np.ones_like(mns_high, dtype=bool)

        cell_size = self.raw_resolution / self.upsample_factor

        # --- 2. PHYSICS CALCS ---
        ndsm = mns_high - mnt_high
        ndsm = np.nan_to_num(ndsm, nan=0)
        
        # STRICT MASKING: We only care about the lot's nDSM
        ndsm = ndsm * lot_mask_high
        
        nx, ny, nz, slope = self.calculate_derivatives(mns_high, cell_size)

        # --- 3. WATERSHED MARKERS ---
        # Foreground: Strict Roof (High, Flat, Smooth)
        roughness = scipy.ndimage.generic_filter(nz, np.std, size=3)
        sure_roof = (ndsm > 1.5) & (slope < 1.0) & (roughness < 0.12) & lot_mask_high
        sure_roof = scipy.ndimage.binary_opening(sure_roof, structure=np.ones((3,3)))
        
        # DEBUG: Store for visualization/stats
        self.last_is_elevated = (ndsm > 1.5)
        self.last_is_roof_slope = (slope < 1.0)
        self.last_is_smooth = (roughness < 0.12)
        self.last_viable_pixels = sure_roof

        # Background: Ground (Low or Steep)
        # We dilate the roof to create a "Safety Zone" (Unknown Region)
        # The watershed will fight inside this zone.
        sure_bg_area = scipy.ndimage.binary_dilation(sure_roof, iterations=5)
        sure_ground = ~sure_bg_area
        
        # Markers for OpenCV (0=Unknown, 1=Ground, 2+=Roofs)
        markers = np.zeros_like(ndsm, dtype=np.int32)
        markers[sure_ground] = 1
        roof_blobs, _ = label(sure_roof)
        markers[roof_blobs > 0] = roof_blobs[roof_blobs > 0] + 1 
        
        # Watershed
        # Normalize nDSM for the algorithm
        ndsm_vis = cv2.normalize(ndsm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ndsm_color = cv2.cvtColor(ndsm_vis, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ndsm_color, markers)
        
        raw_mask = (markers > 1) & lot_mask_high

        # --- 4. [NEW] EDGE REFINEMENT (Regularization) ---
        # Convert the pixelated "Staircase" mask into a Clean Polygon
        final_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        
        # Find contours of the rough mask
        contours, _ = cv2.findContours(raw_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Calculate Perimeter
            peri = cv2.arcLength(cnt, True)
            
            # Douglas-Peucker Approximation
            # epsilon is the "Error Tolerance".
            # 1% of perimeter is usually enough to snap jagged pixels to a straight line
            # without losing the shape of the house.
            epsilon = 0.01 * peri 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Draw the simplified polygon
            cv2.drawContours(final_mask, [approx], -1, 1, thickness=cv2.FILLED)
            
        return final_mask.astype(bool), nx, ny, nz, ndsm

    def calculate_irradiance(self, nx, ny, nz, mask):
        """
        Calculates the 'Solar Score' (Cosine Efficiency).
        """
        # Sun Vector for Montreal (Annual Optimal Average)
        # Approximated as South (Y-) at 45 degree elevation
        # Vector = [0, -1, 1] normalized
        sun = np.array([0, -0.7071, 0.7071])
        
        # Dot Product (Cosine Similarity)
        # How aligned is the surface normal with the sun vector?
        score = (nx * sun[0]) + (ny * sun[1]) + (nz * sun[2])
        
        # Clip negative values (Self-shading / North Face)
        score = np.clip(score, 0, 1)
        
        # Apply mask
        score[~mask] = 0
        
        return score

    def generate_heatmap(self, solar_scores):
        """
        Visualizes the solar potential.
        """
        cmap = cm.get_cmap('inferno')
        rgba = cmap(solar_scores)
        
        # Alpha Channel Logic
        # We want the 'Good' parts to shine, and 'Bad' parts to be ghosted.
        alpha = np.zeros_like(solar_scores)
        
        mask_indices = solar_scores > 0
        # Dynamic Opacity: 0.4 (Low Energy) -> 0.9 (High Energy)
        alpha[mask_indices] = 0.4 + (solar_scores[mask_indices] * 0.5)
        
        rgba[:, :, 3] = alpha
        
        img_bytes = BytesIO()
        plt.imsave(img_bytes, rgba, format='png')
        img_bytes.seek(0)
        return img_bytes
