import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useRef, useState } from 'react';
import Map, {
  type MapRef,
  NavigationControl,
} from 'react-map-gl/maplibre';

// Components
import AddressAutocomplete from './components/AddressAutocomplete';

export default function App() {
  const mapRef = useRef<MapRef>(null);

  // Initial View (Montreal)
  const [viewState, setViewState] = useState({
    longitude: -73.5673,
    latitude: 45.5017,
    zoom: 14,
    pitch: 0,
  });

  // When an address is selected, fly to it
  const handleAddressSelect = (lat: number, lon: number) => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [lon, lat],
        zoom: 19, // Close zoom to see the footprint
        duration: 2000, // Smooth animation
      });
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100vw',
        height: '100vh',
      }}
    >
      {/* --- UI OVERLAY (Search Bar) --- */}
      <div
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          zIndex: 10,
          width: '320px', // Fixed width for the search box
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <div
          style={{
            background: 'white',
            padding: '16px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <h1
            style={{
              margin: '0 0 12px 0',
              fontSize: '1.2rem',
              color: '#333',
            }}
          >
            LumiSol ☀️
          </h1>

          <AddressAutocomplete onSelect={handleAddressSelect} />

          <p
            style={{
              fontSize: '0.75rem',
              color: '#666',
              margin: '8px 0 0 0',
              lineHeight: '1.4',
            }}
          >
            Search for an address to analyze roof potential.
            <br />
            Data provided by Google Maps.
          </p>
        </div>
      </div>

      {/* --- MAP --- */}
      <Map
        ref={mapRef}
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        style={{ width: '100%', height: '100%' }}
        mapLib={maplibregl}
        // Google Maps Style
        mapStyle={{
          version: 8,
          sources: {
            'google-source': {
              type: 'raster',
              tiles: [
                'https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                'https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                'https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
              ],
              tileSize: 256,
              maxzoom: 20,
              attribution: '© Google',
            },
          },
          layers: [
            {
              id: 'satellite-layer',
              type: 'raster',
              source: 'google-source',
              paint: {
                'raster-opacity': 1.0,
                'raster-saturation': 0.1,
              },
            },
          ],
        }}
      >
        <NavigationControl position="top-right" />
      </Map>
    </div>
  );
}
