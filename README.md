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

### Backend (The Brain)
*   **API**: Python (FastAPI)
*   **Spatial Database**: PMTiles / GeoPackage queried via GeoPandas / Shapely
*   **Physics Engine**: Rasterio, NumPy, SciPy

### Algorithms
*   **Super-Resolution**: Bicubic upsampling of 1m LiDAR grids to sub-meter resolution.
*   **Curvature Detection**: Using Laplacian filters to detect "invisible" seams (parapets) between connected buildings.
*   **Vector Math**: Calculating Surface Normals and Solar Incidence angles for every pixel.

## 📐 Key Algorithms

### 1. Lot-Based Isolation (The "Neighbor Firewall")
Solving the segmentation problem by using legal data instead of computer vision.

*   **Input**: User Coordinate (lat, lon).
*   **Process**: Spatial query against the `quebec_lots.pmtiles` database (indexed).
*   **CRS Standardization**: Government LiDAR often uses projected metric coordinates (e.g., MTM Zone 8), while web maps use Lat/Lon (WGS84). The backend automatically reprojects the cadastral polygon (`.to_crs()`) to match the LiDAR's native projection before cropping, ensuring sub-centimeter alignment.
*   **Action**: The LiDAR Digital Surface Model (MNS) is masked. Pixels outside the property line are set to 0.
*   **Benefit**: It is mathematically impossible for the algorithm to "leak" onto a neighbor's roof, even if the roofs are physically connected.

### 2. Super-Resolution & Curvature Barriers
Raw government data is "blocky" (1m/pixel). We treat it as a signal processing problem.

*   **Super-Res**: We upsample the grid by 2x-4x using interpolation to turn "stair-step" roofs into smooth slopes.
*   **Curvature Barrier**: To separate a house from its own extension (or a row-house wall), we calculate the Laplacian Curvature. High curvature indicates a "seam" or wall, which acts as a hard barrier for our segmentation logic.

### 3. Automatic Structure Detection
Instead of a neural network, we use physics-based heuristics:

`Is Structure = (Height > 2.5m) AND (Roughness < Threshold)`

This effectively filters out trees (high roughness) and cars (low height), leaving only viable solar surfaces.

## 🚀 Getting Started

### Prerequisites
*   Node.js (v18+)
*   Python 3.9+
*   Data Requirement: You need the `quebec_lots.pmtiles` file in the `backend/` folder.

### 1. Start the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🔮 Roadmap

*   **Live WCS Connection**: Connect the backend directly to the Ministère des Ressources naturelles et des Forêts (MRNF) WCS server for province-wide LiDAR coverage without local files.
*   **Financial Modeling**: Integrate Hydro-Québec rates and panel efficiency curves to estimate ROI.
*   **Shadow Simulation**: Use ray-marching on the MNS to simulate shadows cast by trees at different times of year (Winter Solstice vs Summer).

---
Built as a Capstone Portfolio Project.