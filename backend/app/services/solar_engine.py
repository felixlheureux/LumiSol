import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import label, generate_binary_structure, zoom, laplace
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO


class SolarEngine:
    def __init__(self):
        # In a real app, these would be WCS URLs.
        # For testing, we assume we have local tiles or a function to fetch them.
        pass

    def calculate_normals(self, elevation_grid, cell_size=1.0):
        """
        Calculates the 3D Normal Vector for every pixel.
        This tells us which way the roof faces (Azimuth) and how steep it is (Slope).
        """
        dy, dx = np.gradient(elevation_grid, cell_size)

        # Normal vector components
        # The normal points "up" and away from the surface
        nx = -dx
        ny = -dy
        nz = 1.0

        # Normalize vectors (make length = 1)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        return nx, ny, nz

    def create_lot_mask(self, shape, polygon_points):
        """
        Creates a binary mask from a list of polygon points (x, y).
        """
        # rasterio.features.rasterize expects a list of (geometry, value) tuples
        # The geometry is a GeoJSON-like dictionary
        geometry = {
            "type": "Polygon",
            "coordinates": [polygon_points] # Note: List of lists of points
        }
        
        mask = features.rasterize(
            [(geometry, 1)],
            out_shape=shape,
            fill=0,
            dtype=np.uint8
        )
        return mask.astype(bool)

    def segment_lot_structures(self, mns, mnt, lot_mask):
        """
        Finds ALL structures on the lot using Connected Component Analysis.
        1. Apply Lot Mask.
        2. Threshold Height > 2.5m (to ignore ground/grass).
        3. Find Blobs (House, Garage, Shed).
        """
        # 1. Apply Lot Mask (Set outside to 0)
        # We assume mns is already normalized (Height above ground)
        # If mns is raw elevation, we need mns - mnt. 
        # But usually 'mns' passed here is the Digital Surface Model.
        # Let's assume input is DSM (mns) and DTM (mnt).
        
        # Calculate Normalized Height (nDSM)
        ndsm = mns - mnt
        ndsm = np.maximum(ndsm, 0) # Remove negative noise
        
        print(f"      [DEBUG] nDSM Max: {np.max(ndsm):.2f}m")
        
        # Mask out neighbors
        if lot_mask is not None:
            print(f"      [DEBUG] Lot Mask Area: {np.sum(lot_mask)} pixels")
            ndsm[~lot_mask] = 0
            
        # 2. Threshold (Height > 2.5m)
        # This removes cars, bushes, fences, etc.
        height_mask = ndsm > 2.5
        
        # 3. Roughness Filter (Remove Trees)
        # Trees have high roughness (variance in height). Roofs are smooth.
        # We use Laplacian as a proxy for roughness.
        roughness = laplace(ndsm)
        roughness = np.abs(roughness)
        
        # Dynamic Thresholding (Isodata Algorithm)
        # We only care about roughness on "potential structures" (Height > 2.5m)
        candidate_pixels = roughness[height_mask]
        
        if len(candidate_pixels) > 0:
            # Initial threshold = mean
            T = np.mean(candidate_pixels)
            for _ in range(10): # Max 10 iterations
                g1 = candidate_pixels[candidate_pixels < T]
                g2 = candidate_pixels[candidate_pixels >= T]
                
                if len(g1) == 0 or len(g2) == 0:
                    break
                    
                m1 = np.mean(g1)
                m2 = np.mean(g2)
                new_T = (m1 + m2) / 2
                
                if abs(new_T - T) < 0.01:
                    T = new_T
                    break
                T = new_T
            
            roughness_threshold = T
            print(f"      [DEBUG] Dynamic Roughness Threshold: {roughness_threshold:.4f}")
        else:
            roughness_threshold = 1.0 # Fallback
            
        roughness_mask = roughness < roughness_threshold
        
        # Combine masks
        structure_binary = height_mask & roughness_mask
        
        print(f"      [DEBUG] Structure Pixels (Height Only): {np.sum(height_mask)}")
        print(f"      [DEBUG] Structure Pixels (Height + Roughness): {np.sum(structure_binary)}")
        
        # 4. Connected Components (Blob Detection)
        # structure defines connectivity (diagonal vs orthogonal)
        s = generate_binary_structure(2, 2) # 8-connectivity
        labeled_array, num_features = label(structure_binary, structure=s)
        
        # Filter small blobs (noise)
        min_size = 10 # pixels (approx 10m^2 if 1m res, or less if super-res)
        # If this is called on high-res data, min_size should be larger
        
        final_mask = np.zeros_like(structure_binary, dtype=bool)
        
        for i in range(1, num_features + 1):
            blob_mask = (labeled_array == i)
            if np.sum(blob_mask) > min_size:
                final_mask |= blob_mask
                
        return final_mask, ndsm

    def calculate_solar_potential(self, mns, mask):
        """
        Simple Physics: Slope + Azimuth = Sun Score
        """
        # Solar heuristic for Quebec (South facing, ~35-45 deg tilt is best)
        # Ideal Aspect: 180 deg (South) -> roughly 3.14 rads or -3.14 depending on coord system
        # Let's simplify: High score if facing South

        # Vector pointing South-ish and Up-ish (The Sun)
        # Roughly representing peak sun position for optimization
        sun_vector = np.array([0, -0.7, 0.7])
        sun_vector /= np.linalg.norm(sun_vector)

        nx, ny, nz = self.calculate_normals(mns)

        # Dot product: How aligned is roof normal with sun vector?
        # Score -1.0 to 1.0
        score = (nx * sun_vector[0]) + \
            (ny * sun_vector[1]) + (nz * sun_vector[2])

        # Clip negative values (North facing / Shadow)
        score = np.clip(score, 0, 1)

        # Apply Building Mask
        score[~mask] = 0

        return score

    def generate_heatmap_overlay(self, solar_scores):
        """
        Generates a transparent PNG image from scores.
        """
        # Create an RGBA image
        # Colormap: 'inferno' (Black -> Red -> Yellow) looks cool for heat
        cmap = cm.get_cmap('inferno')

        # Normalize scores to 0-1 for colormap
        colored_data = cmap(solar_scores)

        # Set Alpha channel:
        # 0.0 (Transparent) where score is 0
        # 0.6 (Semi-transparent) where score is high
        alpha = solar_scores.copy()
        alpha[alpha > 0] = 0.7  # Visible opacity
        colored_data[:, :, 3] = alpha  # Apply to Alpha channel

        # Convert to bytes
        img_bytes = BytesIO()
        plt.imsave(img_bytes, colored_data, format='png')
        img_bytes.seek(0)
        return img_bytes
