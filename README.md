# LumiSol ☀️

LumiSol is an intelligent solar potential analysis engine designed for the Quebec market. It bridges the gap between satellite imagery and physical reality by fusing Official Cadastral Data with LiDAR to perform precise, whole-property energy estimation.

## 🧐 The Problem

Tools like Google Project Sunroof are excellent but have significant limitations:

*   **Coverage Gaps**: They rarely cover rural Quebec, small towns, or specific areas like Sherbrooke effectively.
*   **Missing Structures**: Their data is pre-calculated on main buildings only. If you have a shed, garage, or new extension, it often isn't analyzed.
*   **The "Row House" Paradox**: In cities like Montreal, connected roofs are often merged into one giant blob, making individual analysis impossible for homeowners.

## 💡 The Solution

LumiSol takes a "Legal-Physics Hybrid" approach. Instead of guessing where a building starts and ends using AI vision, we use the **Legal Property Boundary** to isolate the physics simulation.

1.  **The Address Lookup**: The user searches for an address.
2.  **The Legal Firewall**: The backend queries the *Rôle d'évaluation foncière* (Cadastre) to get the exact polygon of the user's lot.
3.  **The Data Isolation**: We crop the raw government LiDAR data (MNS & MNT) to strictly the lot boundaries. This physically deletes neighbors' data from the simulation, solving the "connected roof" issue instantly.
4.  **Whole-Lot Scanning**: Instead of asking the user to click a roof, we run a **Connected Component Analysis** on the isolated lot. This automatically detects every structure (House, Garage, Shed) taller than 2.5m.

## 🛠️ Tech Stack

### Frontend
*   **Framework**: React + Vite (TypeScript)
*   **Mapping**: MapLibre GL JS
*   **Imagery**: Google Satellite / Azure Maps (Bing)
*   **Search**: Address Autocomplete via Photon
*   **Package Manager**: `pnpm`

### Backend (The Brain)
*   **API**: Python (FastAPI)
*   **Manager**: `uv` (Modern Python Package Manager)
*   **Spatial Database**: GeoPackage (`.gpkg`) queried via GeoPandas / Shapely
*   **Physics Engine**: Rasterio, NumPy, SciPy (ndimage)

### Algorithms
*   **Super-Resolution**: Lanczos upsampling (2x) of 1m LiDAR grids to 0.5m resolution. This preserves sharp roof edges better than standard cubic interpolation.
*   **Strict Legal Alignment**: Uses official Cadastral Polygons to mask LiDAR data, ensuring zero leakage into neighboring properties.
*   **Dynamic Roughness Filtering**: Uses Laplacian filters and Isodata thresholding to distinguish smooth roofs from rough trees.
*   **Morphological Cleanup**: Applies Opening and Closing operations to remove noise (salt-and-pepper) and fill small holes in detected structures.

## 📂 Project Structure

```
LumiSol/
├── backend/                 # FastAPI Application
│   ├── app/
│   │   ├── main.py          # Entry Point
│   │   ├── api/             # API Routes
│   │   ├── services/        # Core Logic (SolarEngine, LotManager)
│   │   └── core/            # Config
│   ├── data/                # Data Files (Ignored by Git)
│   │   ├── quebec_lots.gpkg
│   │   └── *.tif
│   ├── scripts/             # Standalone Tools (Debug Visualizer)
│   ├── pyproject.toml       # UV Dependencies
│   └── uv.lock
├── client/                  # React Application
│   ├── src/
│   └── package.json
```

## 🚀 Getting Started

### Prerequisites
*   **Node.js** (v18+) & **pnpm**
*   **Python** (3.13+) & **uv**
*   **Data**: Place `quebec_lots.gpkg` and your LiDAR `.tif` files in `backend/data/`.

### 1. Start the Backend
```bash
cd backend
# Install dependencies and sync environment
uv sync

# Run the API
uv run uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend
```bash
cd client
pnpm install
pnpm dev
```

### 3. Run Visual Debugger
To test the segmentation logic without the frontend:
```bash
cd backend
uv run python scripts/debug_visualizer.py
```
This will generate `debug_output.png` showing the segmentation results for a sample lot.

## 🔮 Roadmap

*   **Live WCS Connection**: Connect the backend directly to the Ministère des Ressources naturelles et des Forêts (MRNF) WCS server for province-wide LiDAR coverage without local files.
*   **Financial Modeling**: Integrate Hydro-Québec rates and panel efficiency curves to estimate ROI.
*   **Shadow Simulation**: Use ray-marching on the MNS to simulate shadows cast by trees at different times of year (Winter Solstice vs Summer).

---
Built as a Capstone Portfolio Project.