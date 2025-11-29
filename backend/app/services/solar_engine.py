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
        OBJECT-BASED SEGMENTATION (OBIA):
        1. Capture ALL high objects (Hysteresis Thresholding).
        2. Classify entire objects as 'Roof' or 'Tree' based on aggregate stats.
        3. 'Shave' the walls at the very end.
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
        
        # Calculate Roughness (Standard Deviation of Z-Normal)
        # We will use this later to judge the whole blob
        roughness_map = scipy.ndimage.generic_filter(nz, np.std, size=3)

        # --- 3. HYSTERESIS THRESHOLDING (Capture the Shape) ---
        # We use two height thresholds to be safe.
        
        # A. CORE (Strong): Definitely a structure (Height > 2.0m)
        core_mask = (ndsm > 2.0) & lot_mask_high
        
        # B. BASIN (Weak): Could be a porch, garage, or edge (Height > 0.8m)
        # We go very low (0.8m) because we will filter out noise later.
        basin_mask = (ndsm > 0.8) & lot_mask_high
        
        # C. RECONSTRUCTION: Grow the Core into the Basin
        # This gives us the "Perfect DSM Shape" you saw, connected to at least one high point.
        # It removes floating trash (bushes) that are only 1m high but have no 2m core.
        structure = generate_binary_structure(2, 2)
        object_mask = scipy.ndimage.binary_dilation(core_mask, mask=basin_mask, iterations=-1, structure=structure)

        # --- 4. OBJECT CLASSIFICATION (The "Brain") ---
        # Now we look at each blob and decide: House or Tree?
        
        labeled_blobs, num_blobs = label(object_mask, structure=structure)
        
        # We will build a new "Clean" mask
        final_mask = np.zeros_like(object_mask)
        
        # Get indices of all pixels
        # We loop through objects (1 to N). This is fast for < 100 objects.
        for i in range(1, num_blobs + 1):
            blob_indices = (labeled_blobs == i)
            
            # --- METRIC 1: Size ---
            # Remove tiny noise (< 2m^2 -> 8 pixels)
            if np.sum(blob_indices) < 8:
                continue
                
            # --- METRIC 2: Roughness ---
            # Calculate the AVERAGE roughness of this entire blob.
            # Trees are chaotic everywhere. Roofs are smooth mostly.
            avg_roughness = np.mean(roughness_map[blob_indices])
            
            # Threshold: 0.12 is a good cutoff. 
            # Trees are usually > 0.20. Roofs are usually < 0.05.
            if avg_roughness > 0.12:
                continue # It's a Tree -> Delete it.
                
            # --- METRIC 3: Wall Artifacts ---
            # If the blob is mostly vertical (Average slope > 1.0 rad), it's a wall/fence.
            avg_slope = np.mean(slope[blob_indices])
            if avg_slope > 1.2:
                continue # It's a Wall -> Delete it.

            # If it passed all tests, it's a Roof. Keep it.
            final_mask[blob_indices] = 1
            
        # DEBUG: Store for visualization/stats
        self.last_is_elevated = object_mask
        self.last_is_roof_slope = (slope < 1.2) # Approximation for debug
        self.last_is_smooth = (roughness_map < 0.12) # Approximation for debug
        self.last_viable_pixels = final_mask

        # --- 5. EDGE POLISHING ---
        # Since we used a low threshold (>0.8m), we captured the walls.
        # We simply erode the mask by 1 pixel (0.5m) to "shave" the walls off.
        final_mask = scipy.ndimage.binary_erosion(final_mask, structure=structure, iterations=1)
        
        # Fill internal holes (Skylights)
        final_mask = scipy.ndimage.binary_fill_holes(final_mask)

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
