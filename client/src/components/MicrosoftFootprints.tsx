import maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';
import { useEffect } from 'react';
import { Layer, Source } from 'react-map-gl/maplibre';

export function MicrosoftFootprints() {
  useEffect(() => {
    // 1. Register the PMTiles protocol globally (once)
    // We check if it's already registered to avoid warnings in React StrictMode
    const protocol = new Protocol();
    maplibregl.addProtocol('pmtiles', protocol.tile);

    return () => {
      maplibregl.removeProtocol('pmtiles');
    };
  }, []);

  // 2. Build the Full URL
  // Vite serves files in 'public/' at the root path.
  // PMTiles requires the full absolute URL (http://...) to handle Range Requests.
  const PMTILES_URL = `pmtiles://${window.location.origin}/footprints.pmtiles`;

  return (
    <Source id="quebec-roofs" type="vector" url={PMTILES_URL}>
      {/* Layer 1: The Orange Fill (Transparent) */}
      <Layer
        id="roofs-fill"
        type="fill"
        source-layer="roofs" // ⚠️ MUST match the '-l' name used in Tippecanoe command
        paint={{
          'fill-color': '#ff9900', // Microsoft Orange
          'fill-opacity': 0.3,
        }}
      />

      {/* Layer 2: The Outline (Crisp edges) */}
      <Layer
        id="roofs-outline"
        type="line"
        source-layer="roofs"
        paint={{
          'line-color': '#ff9900',
          'line-width': 1.5,
          'line-opacity': 0.8,
        }}
      />
    </Source>
  );
}
