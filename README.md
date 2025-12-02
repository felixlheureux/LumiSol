# LumiSol ☀️

**LumiSol** is an industrial-grade Solar Potential Analysis Engine designed for the Canadian market. It replaces traditional heuristic methods with a **State-of-the-Art Deep Learning pipeline** (Mask2Former) and a **Physics-Based Solar Engine** (pvlib) to provide homeowner-specific energy estimates that account for real-world weather, snow loss, and shading.

![LumiSol Architecture](https://via.placeholder.com/800x400?text=LumiSol+AI+Pipeline+Visualization)

## 🚀 Key Features

* **🧠 Next-Gen AI Segmentation**: Uses **Mask2Former** (Swin-Tiny backbone) to detect roof instances with high precision, trained on a fusion of LiDAR Height and Slope data.
* **📐 Geometric Regularization**: A robust post-processing pipeline that "squares" AI predictions into engineering-grade polygons using **Building-Regulariser**, preserving complex L-shapes while enforcing orthogonality.
* **❄️ Winter-Aware Physics**: A custom **Snow Loss Model** calculates energy penalties based on roof slope, temperature, and historical snow depth.
* **📅 Time-Series Simulation**: Instead of generic averages, the engine runs a **365-day hourly simulation** using real historical weather data (ERA5 Reanalysis via Open-Meteo) for the specific location.
* **🍏 Apple Silicon Native**: Optimized training pipeline that runs natively on Mac M2 (MPS) as well as Cloud NVIDIA GPUs (CUDA).

---

## 🛠️ The Tech Stack

### Backend (The Brain)
* **Framework**: Python 3.13 + **FastAPI**
* **AI Core**: PyTorch Lightning + Hugging Face Transformers (`mask2former-swin-tiny`)
* **Geospatial**: Rasterio, GeoPandas, Shapely, PyProj
* **Physics Engine**: `pvlib` (Irradiance), `rvt-py` (Horizon Shading/Sky View Factor)
* **Pipeline Management**: **Typer** (CLI), **uv** (Dependency Management)

### Frontend (The Interface)
* **Framework**: React 19 + Vite (TypeScript)
* **Mapping**: MapLibre GL JS
* **Visualization**: Recharts (Energy Graphs)

---

## 🏗️ Architecture: The 6-Step Pipeline

The backend is organized into a modular pipeline controlled by a central CLI (`lumisol.py`).

| Step | Component | Description |
| :--- | :--- | :--- |
| **1. Filter** | `VectorFilter` | Queries the Government Building Cadastre to identify valid lots and selects optimal LiDAR tiles for training. |
| **2. Download** | `LidarDownloader` | Fetches raw `.tif` LiDAR data (MNS/MNT) from government servers. |
| **3. Generate** | `DataGenerator` | Fuses LiDAR Height, Slope, and Roughness into **512x512 tensors**. Applies **Gaussian Smoothing** to remove sensor noise and upscaling artifacts. |
| **4. Train** | `ModelTrainer` | Fine-tunes **Mask2Former** to recognize roofs in geometric data. Supports dynamic hardware config (`--arch mps/cuda`). |
| **5. Post-Process** | `PostProcessor` | Vectorizes AI rasters, applies **Dilation** (+0.5m) to recover eroded edges, and uses **Building-Regulariser** to enforce clean geometry. |
| **6. Serve** | `GeoEngine` | A real-time API that extracts geometry on-demand and runs the `SolarEngine` physics simulation. |

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.13+** (Installed via `uv` recommended)
* **Node.js 18+** & **pnpm**
* **Hardware**: Apple Silicon (M1/M2/M3) OR NVIDIA GPU (8GB+ VRAM)

### 1. Installation
Clone the repo and install dependencies:

```bash
# Backend
cd backend
uv sync

# Frontend
cd ../client
pnpm install
```

### 2. Run the Data Pipeline (CLI)

You can run the entire data prep and training workflow with one command:

```bash
cd backend
# Run full pipeline (Filter -> Download -> Generate -> Train)
uv run python lumisol.py all

# OR run specific steps
uv run python lumisol.py generate
uv run python lumisol.py train --epochs 20 --batch-size 4 --arch mps
```

### 3. Start the Application

Once the model is trained (models/lumisol_v1), start the servers:

```bash
# Terminal 1: Backend API
cd backend
uv run uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd client
pnpm dev
```

Visit http://localhost:3000, type an address (e.g., in Montreal), and click "Analyze".

## 🔬 Scientific Methodology

### 1. The "Noisy Image" Strategy

To train the AI on 1m resolution LiDAR, we use a False-Color Composite input:

*   **Red Channel**: Normalized Height (nDSM).
*   **Green Channel**: Slope (Gradient Magnitude).
*   **Blue Channel**: Roughness (Laplacian).

> **Note**: A sigma=0.5 Gaussian blur is applied before feature extraction to prevent "stair-step" aliasing from confusing the model.

### 2. Label Noise Handling (Strategy A)

Training masks are eroded by 0.5m to ensure the model learns high-confidence core features and isn't confused by misalignment between Cadastral vectors and LiDAR. During inference, we apply a +0.5m Dilation Buffer to mathematically recover the true roof edge.

### 3. The Solar Physics Engine

We do not use generic "1,000 kWh/kWp" estimates.

*   **Weather**: We fetch hourly TMY data (GHI, DNI, Temp, Snow) for the specific lat/lon.
*   **Shading**: We calculate the Sky View Factor (SVF) from the DSM to account for trees and neighbors.
*   **Snow Loss**: We apply a vectorized loss model:
    *   Loss = 100% if Snow > 2cm AND Temp < 0°C AND Slope < 60°.
    *   Loss = 0% if melting or steep slope.

## 🔮 Roadmap

*   **Commercial Pricing**: Integrate Hydro-Québec rates for ROI calculation.
*   **Battery Sizing**: Use the hourly load profile to recommend battery storage.
*   **Shadow Casting**: Implement full ray-tracing for hourly shadow analysis (Winter Solstice vs Summer).

Built with ❤️ in Montreal.