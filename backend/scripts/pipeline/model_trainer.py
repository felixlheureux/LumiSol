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
import albumentations as A
from transformers import get_polynomial_decay_schedule_with_warmup

# Fix for MPS (Mac) missing operators
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- AUGMENTATION ---
# Geometric transforms only (no color shifts for LiDAR data)
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # ShiftScaleRotate helps the model learn scale invariance
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
])

class SolarInstanceDataset(Dataset):
    """
    Custom Dataset to handle Geometric Tensors (PNG) + Instance Masks (16-bit PNG).
    """
    def __init__(self, img_dir=None, mask_dir=None, processor=None, transform=None, images=None, masks=None):
        if images is not None and masks is not None:
            self.images = images
            self.masks = masks
        elif img_dir is not None and mask_dir is not None:
            self.images = sorted(glob.glob(f"{img_dir}/*.png"))
            self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        else:
            raise ValueError("Must provide either (img_dir, mask_dir) or (images, masks)")
            
        self.processor = processor
        self.transform = transform

        if len(self.images) == 0:
            raise ValueError(f"No images found. Did you run the generator?")

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
        
        # Apply Augmentation
        if self.transform:
            # Albumentations expects 'image' and 'mask'
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 3. Preprocess for Mask2Former
        # We need to map all instance IDs to the "building" class (ID=1)
        # Background (ID=0) maps to 0.
        unique_ids = np.unique(mask)
        instance_id_to_semantic_id = {0: 0}
        for uid in unique_ids:
            if uid == 0: continue
            instance_id_to_semantic_id[uid] = 1
            
        try:
            inputs = self.processor(
                images=image,
                segmentation_maps=mask,
                instance_id_to_semantic_id=instance_id_to_semantic_id,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error processing index {idx}: {self.images[idx]}")
            raise e
        
        # Remove batch dimension added by processor (since we use a DataLoader)
        # Some outputs (like mask_labels) are lists of tensors, others are tensors.
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, list):
                processed_inputs[k] = v[0]
            elif isinstance(v, torch.Tensor):
                processed_inputs[k] = v.squeeze(0)
            else:
                processed_inputs[k] = v
                
        return processed_inputs

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
    def __init__(self, learning_rate=1e-4, model_checkpoint="facebook/mask2former-swin-tiny-coco-instance"):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model_checkpoint = model_checkpoint
        
        # 1. Define Labels
        # We have 2 classes: Background (0) and Building (1)
        self.id2label = {0: "background", 1: "building"}
        
        # 2. Load Pretrained Model
        # We use ignore_mismatched_sizes=True because we are changing the number of classes 
        # from COCO (80 classes) to ours (1 class + background).
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_checkpoint,
            id2label=self.id2label,
            label2id={v: k for k, v in self.id2label.items()},
            ignore_mismatched_sizes=True 
        )
        # Ensure model is in train mode to silence PL warning
        self.model.train()
        
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)
        
        # Calculate total training steps
        if self.trainer.estimated_stepping_batches:
             total_steps = self.trainer.estimated_stepping_batches
        else:
             # Fallback if not available (e.g. old PL version)
             total_steps = 1000 # Dummy fallback
        
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps), # 10% warmup
            num_training_steps=total_steps,
            power=1.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            self.config["MODEL_CHECKPOINT"],
            do_normalize=False,  # CRITICAL: Keep this False for geometric data
            do_resize=False,     # ADD THIS: You already resized in DataGenerator
            do_rescale=True,     # ADD THIS: Rescales 0-255 inputs to 0-1
        )

    def run(self):
        print(f"🚀 Initializing Training Pipeline...")
        print(f"   Model: {self.config['MODEL_CHECKPOINT']}")
        print(f"   Data: {self.config['OUTPUT_DIR']}")

        # 1. Setup Dataset using Config Paths
        img_dir = f"{self.config['OUTPUT_DIR']}/images"
        mask_dir = f"{self.config['OUTPUT_DIR']}/masks"
        
        # We manually split files to apply transforms ONLY to the training set
        all_images = sorted(glob.glob(f"{img_dir}/*.png"))
        all_masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {img_dir}")
            
        # Shuffle and Split (90/10)
        combined = list(zip(all_images, all_masks))
        np.random.seed(self.config.get("RANDOM_SEED", 42))
        np.random.shuffle(combined)
        all_images, all_masks = zip(*combined)
        
        split_idx = int(0.9 * len(all_images))
        train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]
        train_msks, val_msks = all_masks[:split_idx], all_masks[split_idx:]
        
        print(f"   Training Samples: {len(train_imgs)}")
        print(f"   Validation Samples: {len(val_imgs)}")
        
        # Create Datasets
        # Train set GETS transforms
        train_set = SolarInstanceDataset(
            processor=self.processor, 
            transform=train_transform, 
            images=list(train_imgs), 
            masks=list(train_msks)
        )
        
        # Val set gets NO transforms
        val_set = SolarInstanceDataset(
            processor=self.processor, 
            transform=None, 
            images=list(val_imgs), 
            masks=list(val_msks)
        )
        
        # 3. DataLoaders (Dynamic Workers)
        # Use config["NUM_WORKERS"] instead of hardcoded 0
        train_loader = DataLoader(
            train_set, 
            batch_size=self.config["TRAIN_BATCH_SIZE"], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=self.config.get("NUM_WORKERS", 0), # Default to 0 if missing
            persistent_workers=(self.config.get("NUM_WORKERS", 0) > 0) # Optimization for workers > 0
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config["TRAIN_BATCH_SIZE"], 
            collate_fn=collate_fn, 
            num_workers=self.config.get("NUM_WORKERS", 0),
            persistent_workers=(self.config.get("NUM_WORKERS", 0) > 0)
        )
        
        # 4. Initialize Model
        model = SolarMask2Former(
            learning_rate=self.config["LEARNING_RATE"],
            model_checkpoint=self.config["MODEL_CHECKPOINT"]
        )
        
        # 5. Trainer (Dynamic Architecture)
        trainer = pl.Trainer(
            max_epochs=self.config["TRAIN_EPOCHS"],
            
            # 👇 Inject Dynamic Settings Here
            accelerator=self.config.get("ACCELERATOR", "auto"),
            devices=self.config.get("DEVICES", "auto"),
            precision=self.config.get("PRECISION", "32-true"),
            accumulate_grad_batches=self.config.get("ACCUMULATE_BATCHES", 1),
            
            log_every_n_steps=5,
            default_root_dir="logs_mask2former"
        )
        
        print("🚀 Starting Training...")
        trainer.fit(model, train_loader, val_loader)
        
        # 6. Save Model
        output_path = self.config["MODEL_OUTPUT_DIR"]
        print(f"✅ Saving final model to {output_path}...")
        model.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
