import React, { useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import LoginScreen from '../components/LoginScreen';
import HeroScene from '../components/HeroScene';
import TacticalMap from '../components/TacticalMap';
import HUD from '../components/HUD';
import SearchBar from '../components/SearchBar';

function Platform() {
  const { isAuthenticated, is2DView } = useAppContext();

  return (
    <div className="w-screen h-screen bg-tactical-dark relative overflow-hidden">
      {!isAuthenticated && (
        <div className="absolute inset-0 z-50 transition-opacity duration-1000">
          <LoginScreen />
        </div>
      )}

      {/* Main App Content - Only render behind login or when authenticated */}
      <div className={`absolute inset-0 w-full h-full transition-opacity duration-1000 ${isAuthenticated ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        
        {/* 3D WebGL Layer */}
        <div 
          className={`absolute inset-0 w-full h-full transition-opacity duration-300 z-10`}
          style={{ opacity: is2DView ? 0 : 1, pointerEvents: is2DView ? 'none' : 'auto' }}
        >
          <HeroScene />
        </div>

        {/* 2D Map Layer - Preloaded but hidden initially */}
        <div 
          className={`absolute inset-0 w-full h-full transition-opacity duration-500 z-20`}
          style={{ opacity: is2DView ? 1 : 0, pointerEvents: is2DView ? 'auto' : 'none' }}
        >
          <TacticalMap />
        </div>

        {/* UI Overlay (HUD and Search) */}
        <div className="absolute inset-0 z-30 pointer-events-none">
          <HUD />
          <SearchBar />
        </div>

      </div>
    </div>
  );
}

export default Platform;
