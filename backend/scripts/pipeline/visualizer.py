import os
import cv2
import torch
import glob
import random
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from pathlib import Path
from scripts.pipeline.post_processor import PostProcessor

class ModelVisualizer:
    def __init__(self, config):
        self.config = config
        self.post_processor = PostProcessor(config)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"👁️  Loading Model from {self.config['MODEL_OUTPUT_DIR']}...")
        
        # Load the TRAINED model and processor
        try:
            self.processor = Mask2FormerImageProcessor.from_pretrained(
                self.config["MODEL_OUTPUT_DIR"],
                do_normalize=False,
                do_resize=False,
                do_rescale=True
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.config["MODEL_OUTPUT_DIR"]
            ).to(self.device)
            self.model.eval() # Set to evaluation mode (freezes weights)
        except Exception as e:
            print(f"❌ Failed to load model. Did you run 'lumisol.py train' first?")
            raise e

    def run(self, num_samples=5):
        print(f"📸 Generating {num_samples} visual test samples...")
        
        # Setup output dir
        out_dir = self.config["VISUALIZATION_DIR"]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # Find images
        img_dir = os.path.join(self.config["OUTPUT_DIR"], "images")
        mask_dir = os.path.join(self.config["OUTPUT_DIR"], "masks")
        all_images = sorted(glob.glob(f"{img_dir}/*.png"))
        
        if not all_images:
            print("❌ No images found to test.")
            return

        # Pick random samples
        samples = random.sample(all_images, min(num_samples, len(all_images)))
        
        for img_path in samples:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, filename)
            
            # 1. Load Data
            image = cv2.imread(img_path) # BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            # 2. Run Inference
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 3. Post-Process (Convert AI output to Map)
            # Target size is original image size
            target_sizes = [(image.shape[0], image.shape[1])]
            results = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )[0]
            
            # results['segmentation'] is the 2D map of Instance IDs
            instance_map = results["segmentation"].cpu().numpy()
            
            # results['segments_info'] tells us which Class each Instance ID belongs to
            # We want ONLY indices where label_id == 1 ("building")
            # We ignore label_id == 0 ("background")
            building_ids = [
                seg['id'] for seg in results['segments_info'] 
                if seg['label_id'] == 1 
            ]
            
            # Create a binary mask ONLY for buildings
            if building_ids:
                # Set pixel to 1 if its ID is in our list of building_ids
                binary_pred = np.isin(instance_map, building_ids).astype(np.uint8)
            else:
                # If no buildings found, mask is empty
                binary_pred = np.zeros_like(instance_map, dtype=np.uint8)
            
            # Run the Strategy A pipeline
            refined_polys = self.post_processor.process_single_mask(binary_pred)
            
            # Render back to image
            refined_mask_vis = self.post_processor.polygons_to_image(
                refined_polys, 
                (image.shape[0], image.shape[1])
            )
            
            # Colorize the Refined Mask (Green for success)
            refined_vis_color = cv2.applyColorMap(refined_mask_vis, cv2.COLORMAP_DEEPGREEN)
            # Black out the background (applyColorMap makes 0-values colorful sometimes)
            refined_vis_color[refined_mask_vis == 0] = 0
            
            # 5. Visualization Logic (Side-by-Side)
            # A. Input: Show Height Channel (Red) as heatmap
            height_vis = cv2.applyColorMap(image[:,:,2], cv2.COLORMAP_JET) 
            
            # B. Ground Truth: Show unique IDs
            gt_vis = np.zeros_like(height_vis)
            for uid in np.unique(gt_mask):
                if uid == 0: continue
                np.random.seed(int(uid))
                gt_vis[gt_mask == uid] = np.random.randint(50, 255, 3).tolist()
            
            # C. Prediction: Show unique IDs
            pred_vis = np.zeros_like(height_vis)
            # Update Prediction Visualizer to be distinct (Red for Raw/Eroded)
            pred_vis[binary_pred == 1] = [0, 0, 255] 

            # Add Text Labels
            cv2.putText(height_vis, "1. Input (Height)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(gt_vis, "2. Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(pred_vis, "3. AI Raw (Eroded)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(refined_vis_color, "4. AI Refined (+0.5m)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Stitch 4 Panels
            combined = np.hstack([height_vis, gt_vis, pred_vis, refined_vis_color])
            save_path = os.path.join(out_dir, f"compare_{filename.replace('.png', '.jpg')}")
            cv2.imwrite(save_path, combined)
            print(f"   ✨ Saved comparison: {save_path}")

        print(f"✅ Visualization complete. Check {out_dir}")
