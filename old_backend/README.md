LumiSol: Autonomous Geospatial Data Flywheel

This repository documents the Automated Data Engineering Pipeline built to train a state-of-the-art (SOTA) Mask2Former model for rooftop segmentation. Instead of manual labeling, this system synthesizes training data by fusing government vector data with raw LiDAR physics.

## 🏗️ Architecture Overview

The core philosophy is Geometric Deep Learning. We treat buildings as physical objects defined by their 3D shape (Height, Slope, Texture), not just their visual color (RGB). This makes the model invariant to seasons, shadows, and paint color.

### The Pipeline Steps:

1.  **Ingest**: Query provincial indices for LiDAR tiles.
2.  **Stratify**: Select a balanced mix of Urban, Suburban, and Rural tiles.
3.  **Download**: Fetch massive GeoTIFFs (Height Maps) on-demand.
4.  **Engineer Features**: Convert raw Height into a 3-Channel Geometric Tensor.
5.  **Synthesize Labels**: Burn government vector footprints into raster masks.
6.  **Augment**: Apply "Negative Erosion" to fix labeling errors automatically.

## 🚀 The Scripts (Explained)

### 1. `backend/scripts/generate_mask2former_data.py`

"The Engine" - This script orchestrates the data generation pipeline.

**Why it exists**: Deep Learning models need thousands of images. Manually cropping them from a 500GB map is impossible. This script automates the "Cookie Cutting."

**Modular Design**:
The logic is split into reusable components in `backend/scripts/pipeline/`:
-   `VectorFilter`: Handles vector loading, spatial indexing, and stratified sampling.
-   `LidarDownloader`: Manages concurrent downloads of LiDAR tiles.
-   `DataGenerator`: Processes tiles into geometric tensors and masks.

**CLI Usage**:
You can skip specific steps using flags:
```bash
# Run full pipeline
uv run scripts/generate_mask2former_data.py

# Skip vector filtering (if already done)
uv run scripts/generate_mask2former_data.py --skip-vectors

# Skip downloading (if tiles are local)
uv run scripts/generate_mask2former_data.py --skip-download

# Run only generation
uv run scripts/generate_mask2former_data.py --skip-vectors --skip-download
```

**Key Algorithms**:

*   **Stratified Sampling**: It doesn't just pick random tiles. It calculates building density (buildings/km²) and forces a distribution of High Density (Downtown), Medium (Suburbs), and Low (Rural). This prevents the "Empty Forest" bias.
*   **Geometric Feature Engineering**: It calculates `np.gradient()` to create a Slope channel and `scipy.ndimage.laplace()` to create a Roughness channel.
*   **Result**: A 3-channel image (Height, Slope, Roughness) that looks like a "Technical Drawing" to the AI.
*   **Mask Erosion**: It applies a -0.5m buffer to every vector polygon.
*   **Why**: Government data is often "loose" (covering gutters). Erosion forces the AI to learn the safe center of the roof, improving precision at the edges.

### 2. `backend/scripts/train_mask2former.py`

"The Brain" - This script trains the Transformer.

**Why it exists**: To fine-tune a pre-trained Mask2Former (Swin-Tiny backbone) on our custom geometric data.

**Key Techniques**:

*   **Z-Score Normalization**: It calculates (Pixel - Mean) / StdDev for height data. This is crucial for physics data (unlike RGB which is just 0-255).
*   **Boundary IoU**: It optimizes for edge precision, not just blob overlap.
*   **Hugging Face Integration**: It uses the `transformers` library to load SOTA pre-trained weights, saving weeks of training time.

### 3. `backend/scripts/test_generation_local.py`

"The Sanity Check" - A unit test for data quality.

**Why it exists**: To verify that the LiDAR and Vectors align before we burn 10 hours of GPU time.

**Output**: Generates human-readable JPEGs in `debug_vis/` where:
*   **Red/Blue**: Height Heatmap.
*   **Green Overlay**: The vector mask.

If they don't line up, we know we have a CRS projection error.

## 📊 Data Dictionary

### Inputs (Raw Materials)

| Dataset | Source | Purpose | Format |
| :--- | :--- | :--- | :--- |
| **LiDAR Index** | MRNF (Forêt Ouverte) | Locates height tiles across Quebec. | GeoJSON |
| **Building Vectors** | Données Québec (Référentiel) | The "Ground Truth" labels. | GPKG |
| **LiDAR MHC** | MRNF | Raw canopy height (Surface - Ground). | GeoTIFF |

### Outputs (Processed Training Chips)

| File | Format | Content |
| :--- | :--- | :--- |
| `images/tX_bY.png` | PNG (3-Channel) | **Ch 1**: Normalized Height<br>**Ch 2**: Slope (Gradient)<br>**Ch 3**: Roughness (Laplacian) |
| `masks/tX_bY.png` | PNG (16-bit) | **Value 0**: Background<br>**Value 1..N**: Unique Building IDs |

## 🧠 Why This Architecture? (Portfolio Talking Points)

1.  **No "RGB Dependency"**:
    Most pipelines break when clouds cover the satellite photo. This pipeline is blind (it uses LiDAR). It works at night, under clouds, and in winter.

2.  **Self-Healing Data**:
    The Erosion Logic (`.buffer(-0.5)`) automatically fixes the "Jittery Edge" problem common in government data. We built a system that is more accurate than its own training data.

3.  **Infinite Scalability**:
    The system uses Spatial Indexing (`sindex`). It creates training chips by querying the index, not by scanning pixels. This means it runs at O(log N) speed, allowing it to process the entire province of Quebec just as easily as a single neighborhood.

## 🛠️ How to Run

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment
Create `backend/.env` to centralize cache:
```bash
PYTHONPYCACHEPREFIX=.pycache
```

### 3. Generate Data (The Flywheel)
```bash
uv run backend/scripts/generate_mask2former_data.py
```
Watch the progress bar as it crawls the index, downloads tiles, and generates chips.

### 4. Train Model
```bash
uv run backend/scripts/train_mask2former.py
```
This will start the PyTorch Lightning trainer.