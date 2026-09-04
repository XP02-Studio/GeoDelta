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
  const initiateSearch = (query) => {
    setSearchQuery(query);
    
    // Mock Gazetteer: Resolve coordinates based on query
    let newCoords = { lat: 20.5937, lng: 78.9629 }; // fallback
    if (query.toLowerCase().includes('nepal')) {
      newCoords = { lat: 28.3949, lng: 84.1240 };
    } else if (query.toLowerCase().includes('delhi')) {
      newCoords = { lat: 28.7041, lng: 77.1025 };
    } else if (query.toLowerCase().includes('mumbai')) {
      newCoords = { lat: 19.0760, lng: 72.8777 };
    }
    
    // 1. Set target for 3D globe to tween to
    setTargetCoordinates(newCoords);
    setIsAnimatingToTarget(true);

    // 2. The Globe component will handle the GSAP/pointOfView tweening.
    // 3. Once tween completes, we trigger the zoom and then switch to 2D.
    // We will simulate this flow via timeouts in this mock for now, 
    // but ideally the Globe component calls a `onTweenComplete` callback.
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
