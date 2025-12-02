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
  const [layerOpacity, setLayerOpacity] = useState(0.8); // New: Control Heatmap Opacity

  // 3. Analysis State
  const [selectedPoint, setSelectedPoint] = useState<{ lat: number; lon: number } | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  
  const [analysisResult, setAnalysisResult] = useState<{
    heatmap: string; // Full Data URI
    bounds: [[number, number], [number, number]]; // [[w, s], [e, n]]
    solar_potential: string; // "1234 kWh/yr"
    area: string; // "123 m²"
    lot_polygon: [number, number][];
    graph_data: { date: string; kwh: number }[];
  } | null>(null);

  // 4. Dynamic Map Style
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
            'raster-saturation': mapProvider === 'google' ? -0.2 : 0, 
          },
        },
      ],
    };
  }, [mapProvider]);

  // 5. Analysis Logic
  const runAnalysis = useCallback(async (lat: number, lon: number) => {
    setSelectedPoint({ lat, lon });
    setAnalysisResult(null);
    setStatus('loading');
    setErrorMessage('');

    try {
      // Call the Python Backend
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown Error' }));
        throw new Error(errorData.detail || `Error ${response.status}`);
      }

      const data = await response.json();
      setAnalysisResult(data);
      setStatus('success');
    } catch (error: any) {
      console.error('Analysis error:', error);
      setStatus('error');
      setErrorMessage(error.message || 'Could not connect to backend.');
    }
  }, []);

  const handleAddressSelect = (lat: number, lon: number) => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [lon, lat],
        zoom: 19,
        duration: 2000,
      });
    }
    runAnalysis(lat, lon);
  };

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
            <div style={{fontSize: '0.7rem', background: '#eee', padding: '2px 6px', borderRadius: '4px', color: '#666'}}>
                v2.0 Beta
            </div>
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
            {status === 'loading' ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', color: '#ff9900', padding: '20px 0' }}>
                <div className="spinner" />
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Analyzing LiDAR Data...</span>
                <span style={{ fontSize: '0.8rem', color: '#888' }}>Checking height, slope & azimuth</span>
              </div>
            ) : status === 'error' ? (
                <div style={{ color: '#d32f2f', background: '#ffebee', padding: '10px', borderRadius: '8px', fontSize: '0.9rem' }}>
                    <strong>Analysis Failed</strong>
                    <p style={{margin: '5px 0 0 0', fontSize: '0.8rem'}}>{errorMessage}</p>
                </div>
            ) : analysisResult ? (
              <div>
                <h3 style={{ margin: '0 0 5px 0', fontSize: '0.85rem', textTransform: 'uppercase', color: '#888' }}>
                  Solar Potential
                </h3>
                <div style={{ fontSize: '2.2rem', fontWeight: 700, color: '#1a9641', lineHeight: 1 }}>
                  {analysisResult.solar_potential}
                </div>
                
                <div style={{ display: 'flex', gap: '15px', marginTop: '15px', fontSize: '0.85rem' }}>
                  <div>
                    <div style={{ color: '#888' }}>Roof Area</div>
                    <div style={{ color: '#333', fontWeight: 600, fontSize: '1.1rem' }}>{analysisResult.area}</div>
                  </div>
                </div>

                {/* The Graph */}
                <div style={{ marginTop: '20px', height: '180px' }}>
                  <h4 style={{ margin: '0 0 10px 0', fontSize: '0.8rem', color: '#666' }}>
                    Daily Production (Last 365 Days)
                  </h4>
                  
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analysisResult.graph_data}>
                      <XAxis 
                        dataKey="date" 
                        hide={true} // Hide dates to keep it clean
                      />
                      <Tooltip 
                        contentStyle={{ background: '#333', border: 'none', color: '#fff', fontSize: '12px' }}
                        cursor={{fill: 'rgba(0,0,0,0.1)'}}
                        formatter={(val: number) => [`${val} kWh`, 'Energy']}
                        labelFormatter={(label: string) => new Date(label).toLocaleDateString()}
                      />
                      <Bar dataKey="kwh" fill="#ff9900" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Opacity Slider */}
                <div style={{ marginTop: '20px' }}>
                    <div style={{display:'flex', justifyContent:'space-between', fontSize: '0.75rem', marginBottom: '5px', color: '#666'}}>
                        <span>Heatmap Opacity</span>
                        <span>{Math.round(layerOpacity * 100)}%</span>
                    </div>
                    <input 
                        type="range" 
                        min="0" max="1" step="0.1" 
                        value={layerOpacity}
                        onChange={(e) => setLayerOpacity(parseFloat(e.target.value))}
                        style={{width: '100%', accentColor: '#ff9900'}}
                    />
                </div>
              </div>
            ) : (
              <p style={{ fontSize: '0.9rem', color: '#666', lineHeight: '1.4' }}>
                Search an address or click the map to run the Python AI Engine.
              </p>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .spinner {
          width: 24px; height: 24px;
          border-radius: 50%;
          border: 3px solid #ff9900;
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
        mapStyle={mapStyle as any} 
        onClick={onMapClick}
        cursor={status === 'loading' ? 'wait' : 'crosshair'}
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
              url={analysisResult.heatmap}
              coordinates={[
                [analysisResult.bounds[0][0], analysisResult.bounds[1][1]], // Top Left (West, North)
                [analysisResult.bounds[1][0], analysisResult.bounds[1][1]], // Top Right (East, North)
                [analysisResult.bounds[1][0], analysisResult.bounds[0][1]], // Bottom Right (East, South)
                [analysisResult.bounds[0][0], analysisResult.bounds[0][1]], // Bottom Left (West, South)
              ]}
            >
              <Layer
                id="solar-layer"
                type="raster"
                paint={{ 
                  'raster-opacity': layerOpacity,
                  'raster-fade-duration': 0 
                }}
              />
            </Source>

            {/* Optional: Lot Polygon (if supported in future) */}
            {analysisResult.lot_polygon && analysisResult.lot_polygon.length > 0 && (
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
                    'line-width': 2,
                    'line-dasharray': [2, 2]
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