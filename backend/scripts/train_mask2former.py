import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerModel
import evaluate

# --- CONFIG ---
DATA_DIR = "data/processed/mask2former_train"
MODEL_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-instance"
BATCH_SIZE = 2 # Transformers are heavy
LR = 1e-4
NUM_CHANNELS = 5 # RGB + Height + VARI

class SolarInstanceDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.images = sorted(glob.glob(f"{img_dir}/*.tif"))
        self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load 5-Channel Image (TIF)
        # OpenCV loads multi-channel TIFs as-is
        image = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        
        # 2. Load Mask (16-bit PNG)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        
        # 3. Preprocess
        # The processor expects 3 channels (RGB). We need to trick it.
        # We process the RGB part to get the correct size/padding, 
        # then resize the extra channels manually to match.
        
        rgb = image[:, :, :3]
        extra_channels = image[:, :, 3:] # Height + VARI
        
        # Process RGB (Resize, Normalize)
        inputs = self.processor(
            images=rgb,
            segmentation_maps=mask,
            task_inputs=["instance"],
            return_tensors="pt"
        )
        
        # Get the resized RGB tensor: (1, 3, H, W)
        pixel_values = inputs["pixel_values"].squeeze(0) 
        
        # Resize extra channels to match the processor's output size
        # Processor output shape is usually (Height, Width)
        target_h, target_w = pixel_values.shape[1], pixel_values.shape[2]
        
        extra_resized = cv2.resize(
            extra_channels, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize extra channels (already 0-255 uint8, make 0-1 float)
        extra_tensor = torch.from_numpy(extra_resized).permute(2, 0, 1).float() / 255.0
        # Normalize RGB (Processor does mean/std, we stick to that logic or simplistic)
        # Note: Processor output is already normalized.
        
        # Concatenate: (3, H, W) + (2, H, W) -> (5, H, W)
        full_tensor = torch.cat([pixel_values, extra_tensor], dim=0)
        
        return {
            "pixel_values": full_tensor,
            "mask_labels": inputs["mask_labels"][0], # List of tensors
            "class_labels": inputs["class_labels"][0], # List of tensors
            "pixel_mask": inputs["pixel_mask"][0] # Binary mask for padding
        }

def collate_fn(batch):
    # Custom collate because masks are lists of varying length (number of buildings varies)
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    pixel_mask = torch.stack([x["pixel_mask"] for x in batch])
    
    class_labels = [x["class_labels"] for x in batch]
    mask_labels = [x["mask_labels"] for x in batch]
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }

class SolarMask2Former(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # 1. Load Pretrained Model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_CHECKPOINT,
            ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "building"},
            label2id={"background": 0, "building": 1}
        )
        
        # 2. SURGERY: Modify First Layer for 5 Channels
        # The backbone (Swin) input layer is usually `embeddings.patch_embeddings.projection`
        # We need to find the first Conv2d and expand its weights.
        
        # Swin Transformer specific path (verify for other backbones)
        patch_embed = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
        
        old_weights = patch_embed.weight.data # Shape: (EmbedDim, 3, Kernel, Kernel)
        new_weights = torch.zeros(
            (old_weights.shape[0], NUM_CHANNELS, old_weights.shape[2], old_weights.shape[3]),
            device=old_weights.device
        )
        
        # Copy RGB weights
        new_weights[:, :3, :, :] = old_weights
        # Initialize extra channels with average of RGB (better than random)
        avg_weight = torch.mean(old_weights, dim=1, keepdim=True)
        new_weights[:, 3:, :, :] = avg_weight.repeat(1, NUM_CHANNELS-3, 1, 1)
        
        # Replace layer
        patch_embed.weight = torch.nn.Parameter(new_weights)
        patch_embed.in_channels = NUM_CHANNELS
        
        self.metric = evaluate.load("mean_iou")

    def forward(self, pixel_values, mask_labels=None, class_labels=None, pixel_mask=None):
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

if __name__ == "__main__":
    # Setup
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    # Dataset
    train_ds = SolarInstanceDataset(f"{DATA_DIR}/images", f"{DATA_DIR}/masks", processor)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = SolarMask2Former()
    
    # Train
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="auto",
        log_every_n_steps=10
    )
    
    print("🚀 Starting Mask2Former Training (5-Channel)...")
    trainer.fit(model, loader)
    
    # Save
    trainer.save_checkpoint("models/solar_mask2former_final.ckpt")