import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import TrainingArguments, Trainer
import evaluate

# --- CONFIGURATION ---
DATA_DIR = "data/processed/mask2former_lidar_only" # Make sure this matches your generator output
MODEL_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-instance"
OUTPUT_MODEL_DIR = "models/solar_mask2former_lidar"
BATCH_SIZE = 2 
LEARNING_RATE = 1e-4
EPOCHS = 10

class SolarInstanceDataset(Dataset):
    """
    Custom Dataset to handle Geometric Tensors (PNG) + Instance Masks (16-bit PNG).
    """
    def __init__(self, img_dir, mask_dir, processor):
        self.images = sorted(glob.glob(f"{img_dir}/*.png"))
        self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.processor = processor

        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir}. Did you run the generator?")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load 3-Channel Geometric Tensor
        # (Height, Slope, Roughness) saved as PNG
        image = cv2.imread(self.images[idx])
        if image is None:
            raise ValueError(f"Failed to load image: {self.images[idx]}")
            
        # OpenCV loads BGR, convert to RGB order for consistency with pre-trained weights
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Instance Mask (16-bit PNG)
        # 0 = Background, 1..N = Buildings
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load mask: {self.masks[idx]}")
        
        # 3. Preprocess for Mask2Former
        # The processor handles resizing, normalization, and converting instance masks
        # into the binary mask list required by the Transformer.
        try:
            inputs = self.processor(
                images=image,
                segmentation_maps=mask,
                task_inputs=["instance"],
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error processing index {idx}: {self.images[idx]}")
            raise e
        
        # Remove batch dimension added by processor (since we use a DataLoader)
        return {k: v.squeeze(0) for k, v in inputs.items()}

def collate_fn(batch):
    """
    Custom collate because Mask2Former inputs (masks/classes) are lists of varying lengths.
    """
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
    """
    PyTorch Lightning Wrapper for Mask2Former.
    Handles the training loop, optimization, and logging.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Define Labels
        # We have 2 classes: Background (0) and Building (1)
        self.id2label = {0: "background", 1: "building"}
        
        # 2. Load Pretrained Model
        # We use ignore_mismatched_sizes=True because we are changing the number of classes 
        # from COCO (80 classes) to ours (1 class + background).
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_CHECKPOINT,
            id2label=self.id2label,
            label2id={v: k for k, v in self.id2label.items()},
            ignore_mismatched_sizes=True 
        )
        
        # Metric
        self.iou_metric = evaluate.load("mean_iou")

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
        return torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)

def main():
    print(f"🚀 Initializing Training Pipeline...")
    print(f"   Model: {MODEL_CHECKPOINT}")
    print(f"   Data: {DATA_DIR}")

    # 1. Setup Processor
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    # 2. Setup Dataset
    full_dataset = SolarInstanceDataset(f"{DATA_DIR}/images", f"{DATA_DIR}/masks", processor)
    print(f"   Found {len(full_dataset)} training samples.")
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 3. DataLoaders
    # num_workers=0 avoids multiprocessing issues on some systems (Mac/Windows). 
    # Increase to 2 or 4 on Linux/Cloud.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    
    # 4. Initialize Model
    model = SolarMask2Former()
    
    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        accelerator="auto", # Uses GPU (MPS on Mac, CUDA on Nvidia) automatically
        log_every_n_steps=5,
        default_root_dir="logs_mask2former"
    )
    
    print("🚀 Starting Training...")
    trainer.fit(model, train_loader, val_loader)
    
    # 6. Save Model
    print(f"✅ Saving final model to {OUTPUT_MODEL_DIR}...")
    model.model.save_pretrained(OUTPUT_MODEL_DIR)
    processor.save_pretrained(OUTPUT_MODEL_DIR)

if __name__ == "__main__":
    main()