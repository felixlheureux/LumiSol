import numpy as np
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
        FINAL STRATEGY: 'Seed & Grow' with Erosion Correction.
        """
        # --- 1. UPSAMPLING & TERRAIN ---
        mns_high = zoom(mns, self.upsample_factor, order=1)
        mnt_high = self._estimate_terrain(mns_high, window_size_meters=30)
        
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

        # --- 3. THE "SEED" (Safe Center) ---
        roughness = scipy.ndimage.generic_filter(nz, np.std, size=3)
        
        is_high_enough = ndsm > 1.5
        is_smooth_core = roughness < 0.12
        is_flat_core = slope < 1.3
        
        seeds = is_high_enough & is_smooth_core & is_flat_core & lot_mask_high
        
        # DEBUG: Store for visualization/stats
        self.last_is_elevated = is_high_enough
        self.last_is_roof_slope = is_flat_core
        self.last_is_smooth = is_smooth_core
        self.last_viable_pixels = seeds

        # --- 4. THE "POTENTIAL" (The Container) ---
        # 1. Create the base container (Height + Legal Lot)
        potential_mask = (ndsm > 1.5) & lot_mask_high
        
        # 2. THE FIX: Erode the container to remove "interpolated walls"
        # This shaves off the 1-pixel "blur" caused by upsampling/wall-scans.
        # We assume walls are roughly 1-2 pixels thick at this resolution.
        eroded_potential = scipy.ndimage.binary_erosion(potential_mask, iterations=1)

        # --- 5. GROWTH LOOP ---
        structure = generate_binary_structure(2, 2)
        grown_mask = seeds.copy()
        
        for _ in range(40): 
            dilated = scipy.ndimage.binary_dilation(grown_mask, structure=structure)
            
            # Clip growth to the ERODED container
            # This stops the mask BEFORE it slides down the wall
            new_mask = dilated & eroded_potential 
            
            if np.array_equal(new_mask, grown_mask):
                break
            grown_mask = new_mask

        # --- 6. CLEANUP ---
        final_mask = scipy.ndimage.binary_fill_holes(grown_mask)
        final_mask = scipy.ndimage.binary_opening(final_mask, structure=structure, iterations=1)

        return final_mask, nx, ny, nz, ndsm

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
