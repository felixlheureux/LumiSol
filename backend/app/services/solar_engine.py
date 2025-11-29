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

    def segment_solar_facets(self, mns, mnt, lot_mask):
        """
        SCIENTIFIC SEGMENTATION (IMPROVED):
        Optimized for coverage on both Flat (Montreal) and Pitched roofs.
        """
        # 1. Super-Resolution (Bilinear to preserve planar geometry)
        mns_high = zoom(mns, self.upsample_factor, order=1)
        mnt_high = zoom(mnt, self.upsample_factor, order=1)
        
        # Resize lot mask (Nearest Neighbor to keep it sharp boolean)
        if lot_mask is not None and lot_mask.shape != mns_high.shape:
            lot_mask_high = zoom(lot_mask, self.upsample_factor, order=0)
        else:
            lot_mask_high = lot_mask if lot_mask is not None else np.ones_like(mns_high, dtype=bool)

        cell_size = self.raw_resolution / self.upsample_factor

        # 2. Physics Calculations
        nx, ny, nz, slope = self.calculate_derivatives(mns_high, cell_size)
        ndsm = mns_high - mnt_high
        ndsm = np.nan_to_num(ndsm, nan=0)
        
        # 3. DEFINING A "ROOF" (Heuristic Filtering)
        
        # A. Height Filter: Lowered slightly to catch garages/sheds
        is_elevated = ndsm > 2.0 
        
        # B. Slope Filter (CRITICAL FIX): 
        # Allow Flat Roofs (0 deg) up to steep roofs (75 deg = 1.3 rad)
        # We REMOVED the lower bound (> 0.08) because Montreal has flat roofs!
        is_roof_slope = slope < 1.3 
        
        # C. Roughness Filter (Relaxed):
        # We relax the threshold from 0.05 to 0.12 to allow for shingles/chimneys
        roughness = scipy.ndimage.generic_filter(nz, np.std, size=3)
        is_smooth = roughness < 0.12

        # Combine Filters
        viable_pixels = is_elevated & is_roof_slope & is_smooth & lot_mask_high

        # 4. MORPHOLOGICAL CLEANUP (The "Fill" Strategy)
        # You need to import these at the top of your file:
        # from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
        
        # Step A: CLOSING (Connect gaps)
        # This joins pixels that are close together, fixing the "swiss cheese" effect
        structure = generate_binary_structure(2, 2) # 8-connectivity
        closed_mask = scipy.ndimage.binary_closing(viable_pixels, structure=structure, iterations=2)
        
        # Step B: FILL HOLES (Solidify)
        # If we have a ring of valid pixels (e.g., roof edges), fill the inside
        filled_mask = scipy.ndimage.binary_fill_holes(closed_mask)
        
        # Step C: OPENING (Only NOW do we remove noise)
        # Remove tiny specks that are definitely not roofs
        clean_mask = scipy.ndimage.binary_opening(filled_mask, structure=structure, iterations=1)
        
        # 5. CONNECTED COMPONENTS
        labeled_facets, num_features = label(clean_mask)
        
        # Filter tiny fragments (< 1.5m^2)
        # 1 pixel = 0.5 * 0.5 = 0.25m^2. So 6 pixels.
        min_pixels = 6
        final_mask = np.zeros_like(clean_mask)
        
        component_sizes = np.bincount(labeled_facets.ravel())
        if len(component_sizes) > 1:
            valid_labels = np.where(component_sizes > min_pixels)[0]
            valid_labels = valid_labels[valid_labels > 0] # Exclude background
            
            if len(valid_labels) > 0:
                final_mask = np.isin(labeled_facets, valid_labels)

        # 6. Edge Cleanup
        # If the mask spills over the lot line due to 'Closing', re-apply the lot_mask
        final_mask = final_mask & lot_mask_high

        return final_mask, nx, ny, nz

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
