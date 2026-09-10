import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [is2DView, setIs2DView] = useState(false);
  // Default coordinates (India)
  const [targetCoordinates, setTargetCoordinates] = useState({ lat: 20.5937, lng: 78.9629 });
  const [searchQuery, setSearchQuery] = useState('');
  const [isAnimatingToTarget, setIsAnimatingToTarget] = useState(false);

  // Function to handle the complex transition logic
  const initiateSearch = async (query) => {
    setSearchQuery(query);
    
    try {
      // Securely pull API URL from Vite Environment Variables with local fallback
      const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/v1/search/target`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: query, sector_context: "" })
      });

      if (!response.ok) throw new Error("Backend search failed");
      
      const data = await response.json();
      
      // Update coordinates with the data returned from backend (e.g. Sri Lanka)
      setTargetCoordinates({ lat: data.coordinates.lat, lng: data.coordinates.lng });
      setIsAnimatingToTarget(true);

    } catch (error) {
      console.error("API Search Error:", error);
      // Fallback
      setTargetCoordinates({ lat: 20.5937, lng: 78.9629 });
      setIsAnimatingToTarget(true);
    }
  };

  return (
    <AppContext.Provider value={{
      isAuthenticated, setIsAuthenticated,
      is2DView, setIs2DView,
      targetCoordinates, setTargetCoordinates,
      searchQuery, initiateSearch,
      isAnimatingToTarget, setIsAnimatingToTarget
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
