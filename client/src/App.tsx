import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useCallback, useMemo, useRef, useState } from 'react';
import type { MapLayerMouseEvent } from 'react-map-gl/maplibre';
import Map, {
  Layer,
  type MapRef,
  Marker,
  NavigationControl,
  Source,
} from 'react-map-gl/maplibre';

// Components
import AddressAutocomplete from './components/AddressAutocomplete';

// --- MAP PROVIDER CONFIGURATION ---
type ProviderKey = 'google' | 'bing' | 'esri';

const PROVIDERS: Record<ProviderKey, { tiles: string[]; attribution: string; maxzoom: number }> = {
  google: {
    tiles: [
      'https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
      'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
      'https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
      'https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    ],
    attribution: '© Google',
    maxzoom: 20,
  },
  bing: {
    tiles: [
      'https://ecn.t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=129&mkt=en-US&shading=hill&n=z',
      'https://ecn.t1.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=129&mkt=en-US&shading=hill&n=z',
      'https://ecn.t2.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=129&mkt=en-US&shading=hill&n=z',
      'https://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=129&mkt=en-US&shading=hill&n=z',
    ],
    attribution: '© Microsoft',
    maxzoom: 19,
  },
  esri: {
    tiles: [
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    ],
    attribution: '© Esri',
    maxzoom: 18,
  },
};

export default function App() {
  const mapRef = useRef<MapRef>(null);

  // 1. Map View State
  const [viewState, setViewState] = useState({
    longitude: -73.5673,
    latitude: 45.5017,
    zoom: 16,
    pitch: 0,
  });

  // 2. Settings State
  const [mapProvider, setMapProvider] = useState<ProviderKey>('google');

  // 3. Analysis State
  const [selectedPoint, setSelectedPoint] = useState<{ lat: number; lon: number } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<{
    heatmap_b64: string;
    bounds: [number, number, number, number];
    solar_potential: number;
    area_sqm: number;
    lot_polygon: [number, number][];
  } | null>(null);

  // 4. Dynamic Map Style Construction
  const mapStyle = useMemo(() => {
    const current = PROVIDERS[mapProvider];
    return {
      version: 8,
      sources: {
        'satellite-source': {
          type: 'raster',
          tiles: current.tiles,
          tileSize: 256,
          maxzoom: current.maxzoom,
          attribution: current.attribution,
        },
      },
      layers: [
        {
          id: 'satellite-layer',
          type: 'raster',
          source: 'satellite-source',
          paint: {
            'raster-opacity': 1.0,
            'raster-saturation': mapProvider === 'google' ? -0.2 : 0, // Desaturate Google slightly
          },
        },
      ],
    };
  }, [mapProvider]);

  // 5. Shared Analysis Logic
  const runAnalysis = useCallback(async (lat: number, lon: number) => {
    setSelectedPoint({ lat, lon });
    setAnalysisResult(null);
    setIsAnalyzing(true);

    try {
      // Call the Python Backend
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon }),
      });

      if (!response.ok) throw new Error('Analysis failed');
      const data = await response.json();
      setAnalysisResult(data);
    } catch (error) {
      console.error('Error analyzing roof:', error);
      alert('Could not analyze this location. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // 6. Address Selection Handler
  const handleAddressSelect = (lat: number, lon: number) => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [lon, lat],
        zoom: 19,
        duration: 2000,
      });
    }
    // Auto-trigger analysis
    runAnalysis(lat, lon);
  };

  // 7. Map Click Handler
  const onMapClick = useCallback((event: MapLayerMouseEvent) => {
    const { lng, lat } = event.lngLat;
    runAnalysis(lat, lng);
  }, [runAnalysis]);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      
      {/* --- UI SIDEBAR --- */}
      <div
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          zIndex: 10,
          width: '340px',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <div
          style={{
            background: 'white',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <h1 style={{ margin: 0, fontSize: '1.4rem', color: '#333', fontWeight: 800 }}>
              LumiSol <span style={{ color: '#ff9900' }}>Live</span>
            </h1>
          </div>

          <AddressAutocomplete onSelect={handleAddressSelect} />

          {/* MAP PROVIDER TOGGLE */}
          <div style={{ marginTop: '15px', display: 'flex', gap: '8px', background: '#f0f0f0', padding: '4px', borderRadius: '8px' }}>
            {(['google', 'bing', 'esri'] as ProviderKey[]).map((key) => (
              <button
                key={key}
                onClick={() => setMapProvider(key)}
                style={{
                  flex: 1,
                  padding: '6px 0',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                  background: mapProvider === key ? 'white' : 'transparent',
                  color: mapProvider === key ? '#333' : '#888',
                  boxShadow: mapProvider === key ? '0 2px 4px rgba(0,0,0,0.1)' : 'none',
                  transition: 'all 0.2s ease'
                }}
              >
                {key}
              </button>
            ))}
          </div>

          {/* DYNAMIC RESULTS SECTION */}
          <div style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '15px' }}>
            {isAnalyzing ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#ff9900' }}>
                <div className="spinner" />
                <span style={{ fontWeight: 600 }}>Scanning Lot...</span>
              </div>
            ) : analysisResult ? (
              <div>
                <h3 style={{ margin: '0 0 5px 0', fontSize: '0.85rem', textTransform: 'uppercase', color: '#888' }}>
                  Total Lot Potential
                </h3>
                <div style={{ fontSize: '2.2rem', fontWeight: 700, color: '#1a9641', lineHeight: 1 }}>
                  {analysisResult.solar_potential.toLocaleString()} <span style={{ fontSize: '1rem', color: '#666' }}>kWh</span>
                </div>
                
                <div style={{ display: 'flex', gap: '15px', marginTop: '15px', fontSize: '0.85rem' }}>
                  <div>
                    <div style={{ color: '#888' }}>Total Area</div>
                    <div style={{ fontWeight: 600 }}>{analysisResult.area_sqm} m²</div>
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ fontSize: '0.9rem', color: '#666', lineHeight: '1.4' }}>
                Select a provider and search an address to analyze.
              </p>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .spinner {
          width: 18px; height: 18px;
          border-radius: 50%;
          border: 2px solid #ff9900;
          border-top-color: transparent;
          animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      {/* --- MAP --- */}
      <Map
        ref={mapRef}
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        style={{ width: '100%', height: '100%' }}
        mapLib={maplibregl}
        mapStyle={mapStyle as any} // Cast to any to avoid strict type issues with dynamic styles
        onClick={onMapClick}
        cursor={isAnalyzing ? 'wait' : 'crosshair'}
      >
        <NavigationControl position="top-right" />

        {/* Visual Feedback: Where did I click? */}
        {selectedPoint && (
          <Marker longitude={selectedPoint.lon} latitude={selectedPoint.lat} color="#ff9900" />
        )}

        {/* The Result: AI Heatmap Overlay */}
        {analysisResult && (
          <>
            <Source
              id="solar-result"
              type="image"
              url={analysisResult.heatmap_b64}
              coordinates={[
                [analysisResult.bounds[0], analysisResult.bounds[3]], // Top Left
                [analysisResult.bounds[2], analysisResult.bounds[3]], // Top Right
                [analysisResult.bounds[2], analysisResult.bounds[1]], // Bottom Right
                [analysisResult.bounds[0], analysisResult.bounds[1]], // Bottom Left
              ]}
            >
              <Layer
                id="solar-layer"
                type="raster"
                paint={{ 
                  'raster-opacity': 0.8,
                  'raster-fade-duration': 0 
                }}
              />
            </Source>

            {/* LOT POLYGON OVERLAY */}
            {analysisResult.lot_polygon && (
              <Source
                id="lot-polygon"
                type="geojson"
                data={{
                  type: 'Feature',
                  geometry: {
                    type: 'Polygon',
                    coordinates: [analysisResult.lot_polygon],
                  },
                  properties: {},
                }}
              >
                <Layer
                  id="lot-line"
                  type="line"
                  paint={{
                    'line-color': '#ff9900',
                    'line-width': 3,
                  }}
                />
              </Source>
            )}
          </>
        )}
      </Map>
    </div>
  );
}