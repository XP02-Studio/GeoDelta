import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [is2DView, setIs2DView] = useState(false);
  // Default coordinates (India)
  const [targetCoordinates, setTargetCoordinates] = useState({ lat: 20.5937, lng: 78.9629 });
  const [targetBounds, setTargetBounds] = useState(null);
  const [liveSearch, setLiveSearch] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchError, setSearchError] = useState('');
  const [isAnimatingToTarget, setIsAnimatingToTarget] = useState(false);

  // Function to handle the complex transition logic
  const initiateSearch = async (query) => {
    setSearchQuery(query);
    setSearchError('');
    
    try {
      // Securely pull API URL from Vite Environment Variables with local fallback
      const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/v1/search/target`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query_text: query, top_k: 5 })
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Backend search failed");
      }
      
      const data = await response.json();
      
      const lat = data.coordinates?.lat ?? data.results?.[0]?.lat;
      const lng = data.coordinates?.lng ?? data.results?.[0]?.lng;
      if (lat == null || lng == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lng))) {
        throw new Error("Search returned no geographic coordinates");
      }

      setTargetCoordinates({ lat: Number(lat), lng: Number(lng) });

      const bbox = data.bbox ?? data.results?.[0]?.geometry?.coordinates?.[0];
      if (Array.isArray(bbox) && bbox.length >= 4) {
        let west, south, east, north;
        if (data.bbox) {
          [west, south, east, north] = bbox.map(Number);
        } else {
          const lngs = bbox.map(c => c[0]);
          const lats = bbox.map(c => c[1]);
          west = Math.min(...lngs); east = Math.max(...lngs);
          south = Math.min(...lats); north = Math.max(...lats);
        }
        setTargetBounds([[south, west], [north, east]]);
      }

      setLiveSearch(data);
      setIsAnimatingToTarget(true);

    } catch (error) {
      console.error("API Search Error:", error);
      setSearchError(error.message || "Search is unavailable");
    }
  };

  return (
    <AppContext.Provider value={{
      isAuthenticated, setIsAuthenticated,
      is2DView, setIs2DView,
      targetCoordinates, setTargetCoordinates, targetBounds,
      liveSearch,
      searchQuery, searchError, initiateSearch,
      isAnimatingToTarget, setIsAnimatingToTarget
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
