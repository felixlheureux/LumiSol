import numpy as np
import cv2
import scipy.ndimage
from scipy.ndimage import zoom, label, generate_binary_structure, gaussian_filter, binary_opening, binary_closing, binary_fill_holes
from matplotlib import cm
from io import BytesIO
import matplotlib.pyplot as plt
from rasterio.mask import mask as rio_mask
from rasterio import features
from samgeo import SamGeo
import os
from segment_anything import sam_model_registry, SamPredictor

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

    def generate_shadowless_texture(self, array):
        """
        Creates a 'Visual Texture' map for the AI that has NO SHADOWS.
        Instead of simulating a Sun (which casts shadows), we visualize
        Slope and Roughness directly.
        """
        # 1. Calculate Slope (The steepness)
        x, y = np.gradient(array)
        slope = np.sqrt(x*x + y*y)
        
        # 2. Normalize Slope for Visualization (0-255)
        # Steep walls become bright, flat roofs become dark/gray.
        # This creates distinct boundaries for the AI without casting shadows.
        slope_norm = np.clip(slope, 0, 1.5) / 1.5 * 255
        
        # 3. Calculate Curvature/Roughness (The edges)
        # This highlights the "outline" of the roof
        curvature = scipy.ndimage.laplace(array)
        curv_norm = np.clip(np.abs(curvature), 0, 1.0) * 255
        
        # 4. Combine into a "Texture Map"
        # We blend them to give the AI a rich shape to look at.
        texture = (0.7 * slope_norm + 0.3 * curv_norm).astype(np.uint8)
        
        return texture

    def generate_hillshade(self, array, azimuth=315, angle_altitude=45):
        """
        Converts a DSM (Height Map) into a Visual Hillshade (Image).
        This makes the 'Texture' of trees vs roofs visible to the AI.
        """
        x, y = np.gradient(array)
        slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.
        altituderad = angle_altitude * np.pi / 180.
         
        shaded = np.sin(altituderad) * np.sin(slope) + \
                 np.cos(altituderad) * np.cos(slope) * \
                 np.cos(azimuthrad - aspect)
        
        # Normalize to 0-255 (Grayscale Image)
        return (255 * (shaded + 1) / 2).astype(np.uint8)

    def segment_solar_facets(self, mns, mnt, lot_mask, transform):
        """
        SHADOW-PROOF AI SEGMENTATION:
        1. Input: 'Shadowless' Texture Map (Slope/Edge visualization).
        2. Prompt: Peak of the roof.
        3. Constraint: Result MUST be elevated (> 0.5m).
        """
        # --- 1. PRE-PROCESSING ---
        # Upsample 1m -> 0.5m for better resolution
        mns_high = zoom(mns, self.upsample_factor, order=1)
        # Use simple terrain resize (we trust input MNT or use a flat plane if needed)
        mnt_high = zoom(mnt, self.upsample_factor, order=1)
        
        # Resize Lot Mask
        if lot_mask is not None:
             # Nearest Neighbor for boolean mask
            if lot_mask.shape == mns_high.shape:
                lot_mask_high = lot_mask
            else:
                lot_mask_high = zoom(lot_mask, self.upsample_factor, order=0)
        else:
            lot_mask_high = np.ones_like(mns_high, dtype=bool)

        cell_size = self.raw_resolution / self.upsample_factor

        # --- 2. PHYSICS ---
        ndsm = mns_high - mnt_high
        ndsm = np.nan_to_num(ndsm, nan=0)
        
        # STRICT MASKING: We only care about the lot's nDSM
        ndsm = ndsm * lot_mask_high
        
        nx, ny, nz, slope = self.calculate_derivatives(mns_high, cell_size)
        
        # --- 3. AI SEGMENTATION ---
        
        # 1. Prepare the Visual Input (Shadowless)
        # We stop using 'hillshade' which creates fake shadows.
        # We use a Slope/Edge map which looks like a technical drawing.
        texture_gray = self.generate_shadowless_texture(mns_high)
        
        # 2. Find the "Prompt" (The Seed)
        # We don't need a perfect segmentation, just ONE valid pixel.
        # Find the highest point in the lot (most likely the roof peak)
        valid_area = (ndsm > 2.0)
        
        if np.sum(valid_area) == 0:
            # Fallback: No house found
            return np.zeros_like(mns_high, dtype=bool), nx, ny, nz, ndsm, texture_gray
            
        # Get coordinates of the "Center of Mass" of the high area
        y_indices, x_indices = np.where(valid_area)
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        # 3. Run Segment Anything (SAM)
        # Use direct SAM predictor to avoid wrapper issues
        checkpoint_path = "/Users/felix/.cache/torch/hub/checkpoints/sam_vit_b_01ec64.pth"
        if not os.path.exists(checkpoint_path):
             # Fallback if not found (should be there from previous run)
             raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        predictor = SamPredictor(sam)
        
        # SAM expects RGB image (Duplicate channels)
        texture_rgb = np.dstack((texture_gray, texture_gray, texture_gray))
        predictor.set_image(texture_rgb)
        
        # Predicting with a Point Prompt:
        # Note: point_coords expects [[x, y]] where x=col, y=row
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[center_x, center_y]]), # The "Click" (Pixel Coords)
            point_labels=np.array([1]), # 1 = Foreground
            multimask_output=False # We want the best single mask
        )
        
        # 4. Cleanup
        # SAM returns masks of shape (1, H, W) if multimask_output=False
        mask = masks[0]
            
        # Constraint A: Must be inside the Lot
        mask = mask & lot_mask_high
        
        # Constraint B: Must be ELEVATED.
        # Shadows are 2D images projected on the ground (Height ~ 0).
        # We set a low bar (0.5m) to keep porches, but kill ground shadows.
        is_elevated = ndsm > 0.5
        
        final_mask = mask & is_elevated
        
        # DEBUG: Store for visualization/stats
        # We calculate these just for the debugger, even if AI doesn't use them directly
        roughness = scipy.ndimage.generic_filter(nz, np.std, size=3)
        
        self.last_is_elevated = (ndsm > 2.0)
        self.last_is_roof_slope = (slope < 1.0) # Standard roof slope
        self.last_is_smooth = (roughness < 0.15) # Standard roof roughness
        self.last_viable_pixels = final_mask

        return final_mask.astype(bool), nx, ny, nz, ndsm, texture_gray

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
        
        # SMOOTHING: Apply Gaussian Filter to "iron out" the bumps
        # sigma=1.0 is a gentle smooth (approx 3x3 pixel window)
        score = gaussian_filter(score, sigma=1.0)
        
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
