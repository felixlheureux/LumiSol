import debounce from 'lodash/debounce';
import { useEffect, useMemo, useRef, useState } from 'react';

type Suggestion = {
  display_name: string;
  lat: number;
  lon: number;
};

type Props = {
  onSelect: (lat: number, lon: number) => void;
};

export default function AddressAutocomplete({ onSelect }: Props) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // A ref to track if we should skip the next search
  const skipSearchRef = useRef(false);

  const fetchSuggestions = async (searchQuery: string) => {
    if (!searchQuery || searchQuery.length < 3) {
      setSuggestions([]);
      return;
    }

    setIsLoading(true);

    try {
      // Using Photon API (OpenStreetMap data) limited to Montreal area approx
      const url = `https://photon.komoot.io/api/?q=${encodeURIComponent(
        searchQuery
      )}&lat=45.5&lon=-73.5&limit=5&lang=en`;

      const res = await fetch(url);
      const data = await res.json();

      const formatted = data.features.map((f: any) => {
        const { name, housenumber, street, city, state, country } =
          f.properties;
        const streetLine = [housenumber, street]
          .filter(Boolean)
          .join(' ');
        const mainName = name ? `${name}, ${streetLine}` : streetLine;
        const displayName = [mainName, city, state, country]
          .filter((val) => val && val.trim().length > 0)
          .join(', ');

        return {
          display_name: displayName || 'Unknown Location',
          lat: f.geometry.coordinates[1],
          lon: f.geometry.coordinates[0],
        };
      });

      setSuggestions(formatted);
      setIsOpen(true);
    } catch (error) {
      console.error('Geocoding error:', error);
      setSuggestions([]);
    } finally {
      setIsLoading(false);
    }
  };

  const debouncedFetch = useMemo(
    () => debounce((q: string) => fetchSuggestions(q), 400),
    []
  );

  useEffect(() => {
    if (skipSearchRef.current) {
      skipSearchRef.current = false;
      return;
    }

    debouncedFetch(query);
    return () => debouncedFetch.cancel();
  }, [query, debouncedFetch]);

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        marginBottom: '10px',
      }}
    >
      <div style={{ position: 'relative' }}>
        <input
          type="text"
          value={query}
          placeholder="Search address..."
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (query.length > 2 && suggestions.length > 0)
              setIsOpen(true);
          }}
          onBlur={() => setTimeout(() => setIsOpen(false), 200)}
          style={{
            width: '100%',
            padding: '10px 12px',
            paddingRight: '35px',
            fontSize: '14px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            outline: 'none',
            color: '#333',
            boxSizing: 'border-box',
          }}
        />

        {isLoading && (
          <div
            style={{
              position: 'absolute',
              right: '10px',
              top: '50%',
              transform: 'translateY(-50%)',
              width: '14px',
              height: '14px',
              border: '2px solid #e2e8f0',
              borderTop: '2px solid #666',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }}
          />
        )}
      </div>

      {isOpen && suggestions.length > 0 && (
        <ul
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            background: 'white',
            border: '1px solid #e2e8f0',
            borderRadius: '4px',
            listStyle: 'none',
            padding: '0',
            marginTop: '4px',
            zIndex: 9999,
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            color: 'black',
            textAlign: 'left',
            maxHeight: '200px',
            overflowY: 'auto',
          }}
        >
          {suggestions.map((item, index) => (
            <li
              key={index}
              onClick={() => {
                debouncedFetch.cancel();
                skipSearchRef.current = true;
                setQuery(item.display_name);
                setSuggestions([]);
                setIsOpen(false);
                onSelect(item.lat, item.lon);
              }}
              style={{
                padding: '10px 12px',
                cursor: 'pointer',
                borderBottom: '1px solid #f1f5f9',
                fontSize: '13px',
                color: '#1e293b',
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = '#f8fafc')
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = 'white')
              }
            >
              {item.display_name}
            </li>
          ))}
        </ul>
      )}
      <style>{`@keyframes spin { 0% { transform: translateY(-50%) rotate(0deg); } 100% { transform: translateY(-50%) rotate(360deg); } }`}</style>
    </div>
  );
}
