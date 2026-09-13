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
      
      const bestMatch = data.results?.[0];
      if (!bestMatch) {
        throw new Error("Search returned no results");
      }

      const lat = bestMatch.lat ?? bestMatch.coordinates?.lat;
      const lng = bestMatch.lng ?? bestMatch.coordinates?.lng;
      if (lat == null || lng == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lng))) {
        throw new Error("Search returned no geographic coordinates");
      }

      setTargetCoordinates({ lat: Number(lat), lng: Number(lng) });

      if (bestMatch.geometry) {
        const coords = bestMatch.geometry.coordinates?.[0];
        if (Array.isArray(coords) && coords.length >= 4) {
          const lngs = coords.map(c => c[0]);
          const lats = coords.map(c => c[1]);
          const west = Math.min(...lngs);
          const east = Math.max(...lngs);
          const south = Math.min(...lats);
          const north = Math.max(...lats);
          setTargetBounds([[south, west], [north, east]]);
        }
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
