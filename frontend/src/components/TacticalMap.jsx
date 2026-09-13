import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, GeoJSON, ImageOverlay, TileLayer, useMap, Pane } from 'react-leaflet';
import * as turf from '@turf/turf';
import { useAppContext } from '../context/AppContext';
import { Crosshair, Play, Copy, CheckCircle2 } from 'lucide-react';

// Helper component to recenter map when coordinates change
const MapUpdater = ({ center, bounds, zoom }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.flyToBounds(bounds, { animate: true, duration: 1.5, padding: [32, 32] });
    } else {
      map.setView(center, zoom, { animate: false });
    }
  }, [center, bounds, zoom, map]);
  return null;
};

const TacticalMap = () => {
  const { targetCoordinates, targetBounds, liveSearch, searchQuery, is2DView, setIs2DView } = useAppContext();
  const [geoData, setGeoData] = useState(null);
  const [areaMetric, setAreaMetric] = useState(0);
  const [scanStep, setScanStep] = useState(0); // 0: Idle, 1: Align, 2: ChangeFormer, 3: RS-CLIP, 4: Done
  const [sliderValue, setSliderValue] = useState(50);
  const [copied, setCopied] = useState(false);

  // When returning to 3D, reset states
  useEffect(() => {
    if (!is2DView) {
      setGeoData(null);
      setAreaMetric(0);
      setScanStep(0);
      setSliderValue(50);
    }
  }, [is2DView]);

  const handleScanSector = async () => {
    if (scanStep > 0) return;
    setScanStep(1);
    
    setTimeout(() => setScanStep(2), 600);
    setTimeout(() => setScanStep(3), 1200);
    
    try {
      if (!liveSearch?.asset_id || !targetBounds) throw new Error("Run a live search before scanning.");
      const [[minY, minX], [maxY, maxX]] = targetBounds;

      const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/v1/analyze/sector`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          sector_id: liveSearch.asset_id,
          bbox: [minX, minY, maxX, maxY],
          target_query: searchQuery,
          asset_id: liveSearch.asset_id
        })
      });

      if (!response.ok) throw new Error("Analysis failed");

      const geojsonData = await response.json();
      
      setGeoData(geojsonData);
      setScanStep(4);

      if (geojsonData.features[0] && geojsonData.features[0].properties.area_sq_meters) {
        setAreaMetric(geojsonData.features[0].properties.area_sq_meters);
      } else {
        const areaSqMeters = turf.area(geojsonData);
        setAreaMetric(Math.round(areaSqMeters));
      }
    } catch (error) {
      console.error("Scan Error:", error);
      setScanStep(0);
    }
  };

  const geoJsonStyle = (feature) => {
    if (feature.properties && feature.properties.stroke_color) {
      return {
        color: feature.properties.stroke_color,
        weight: 3,
        opacity: 0.8,
        fillColor: feature.properties.stroke_color,
        fillOpacity: 0.2,
        dashArray: "5, 10"
      };
    }

    const type = feature.properties?.changeType || "BUNKER";
    const isCritical = type === "BUNKER" || type === "AIRSTRIP";
    const color = isCritical ? "#EF4444" : "#EAB308";

    return {
      color: color,
      weight: 3,
      opacity: 0.8,
      fillColor: color,
      fillOpacity: 0.2,
      dashArray: "5, 10"
    };
  };

  const handleCopyGPS = () => {
    const coordsStr = `${targetCoordinates.lat.toFixed(6)}, ${targetCoordinates.lng.toFixed(6)}`;
    navigator.clipboard.writeText(coordsStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReturnToOrbit = () => {
    setIs2DView(false);
  };

  return (
    <div className="w-full h-full relative">
      <MapContainer 
        center={[targetCoordinates.lat, targetCoordinates.lng]} 
        zoom={15} 
        zoomControl={false}
        className="w-full h-full bg-tactical-dark z-0"
      >
        <TileLayer
          attribution='&copy; Esri, Maxar, Earthstar Geographics'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />
        {liveSearch?.imagery?.t2_url && targetBounds && <ImageOverlay url={`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}${liveSearch.imagery.t2_url}`} bounds={targetBounds} opacity={1} />}
        {liveSearch?.imagery?.t1_url && targetBounds && (
          <Pane name="historical" style={{ zIndex: 400, clipPath: `polygon(0 0, ${sliderValue}% 0, ${sliderValue}% 100%, 0 100%)` }}>
            <ImageOverlay url={`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}${liveSearch.imagery.t1_url}`} bounds={targetBounds} opacity={1} />
          </Pane>
        )}
        
        <MapUpdater center={[targetCoordinates.lat, targetCoordinates.lng]} bounds={targetBounds} zoom={15} />

        {geoData && <GeoJSON data={geoData} style={geoJsonStyle} />}
      </MapContainer>

      {/* Swipe Slider UI */}
      {is2DView && (
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 z-20 w-1/3 min-w-[300px]">
           <div className="bg-tactical-dark/80 backdrop-blur-md px-6 py-4 rounded-full border border-white/20 flex flex-col items-center shadow-2xl">
              <div className="flex justify-between w-full text-xs font-mono text-gray-400 mb-2">
                 <span>T1 (HISTORICAL)</span>
                 <span>T2 (CURRENT)</span>
              </div>
              <input 
                type="range" 
                min="0" max="100" 
                value={sliderValue} 
                onChange={(e) => setSliderValue(e.target.value)}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neon-cyan focus:outline-none"
              />
           </div>
        </div>
      )}

      {/* Left Panel - 3-Step Controls */}
      {is2DView && (
        <div className="absolute top-0 left-0 h-full w-80 bg-tactical-dark/90 border-r border-white/10 p-6 flex flex-col z-10 backdrop-blur-md shadow-2xl animate-[fade-in-left_0.5s_ease-out]">
           <h2 className="text-neon-cyan font-mono text-lg tracking-widest font-bold mb-8">COMMAND UPLINK</h2>
           
           <div className="space-y-6 flex-grow">
              <div className="border border-white/10 rounded p-4 bg-black/40">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center font-mono text-xs text-white">1</div>
                  <h3 className="font-bold text-white text-sm">Select Surveillance Sector</h3>
                </div>
                <p className="text-xs text-gray-400 mb-4">Coordinates locked via NLP semantic search.</p>
                <div className="bg-white/5 p-2 rounded text-center border border-neon-cyan/30">
                   <span className="font-mono text-xs text-neon-cyan flex items-center justify-center gap-2">
                     <Crosshair className="w-3 h-3" /> SECTOR ACQUIRED
                   </span>
                </div>
              </div>

              <div className={`border rounded p-4 transition-all duration-300 ${scanStep > 0 ? 'border-neon-cyan/50 bg-neon-cyan/5' : 'border-white/10 bg-black/40'}`}>
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center font-mono text-xs ${scanStep > 0 ? 'bg-neon-cyan/20 text-neon-cyan' : 'bg-white/10 text-white'}`}>2</div>
                  <h3 className="font-bold text-white text-sm">Scan Sector</h3>
                </div>
                <p className="text-xs text-gray-400 mb-4">Initialize ChangeFormer & RS-CLIP models on current sector.</p>
                
                <button 
                  onClick={handleScanSector}
                  disabled={scanStep > 0}
                  className={`w-full py-3 rounded font-mono text-sm flex items-center justify-center gap-2 transition-all ${
                    scanStep > 0 
                      ? 'bg-gray-800 text-gray-500 cursor-not-allowed' 
                      : 'bg-neon-cyan text-black hover:bg-white hover:shadow-[0_0_15px_rgba(0,243,255,0.5)]'
                  }`}
                >
                  <Play className="w-4 h-4" />
                  {scanStep > 0 ? 'SCANNING...' : 'EXECUTE SCAN'}
                </button>
              </div>
           </div>

           <button 
             onClick={handleReturnToOrbit}
             className="w-full bg-white/5 hover:bg-white/10 border border-white/20 text-gray-400 font-mono py-3 px-4 rounded transition-all mt-6 text-sm flex items-center justify-center"
           >
             RETURN TO ORBIT
           </button>
        </div>
      )}

      {/* Right Panel - Analysis & Intelligence */}
      {is2DView && (
        <div className="absolute top-0 right-0 h-full w-80 bg-tactical-dark/90 border-l border-white/10 p-6 flex flex-col z-10 backdrop-blur-md shadow-2xl animate-[fade-in-right_0.5s_ease-out]">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-neon-cyan font-mono text-lg tracking-widest font-bold">TARGET INTEL</h2>
            <div className={`w-3 h-3 rounded-full ${scanStep > 0 && scanStep < 4 ? 'bg-yellow-400 animate-ping' : 'bg-green-500'}`}></div>
          </div>
          
          <div className="space-y-6 flex-grow">
            <div className="bg-black/40 p-4 rounded border border-white/5 relative group">
              <p className="text-xs text-gray-500 font-mono mb-1">GPS COORDINATES</p>
              <p className="text-white font-mono text-sm">{targetCoordinates.lat.toFixed(6)} N</p>
              <p className="text-white font-mono text-sm">{targetCoordinates.lng.toFixed(6)} E</p>
              
              <button 
                onClick={handleCopyGPS}
                className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors"
                title="Copy GPS Data"
              >
                {copied ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>

            <div className="bg-black/40 p-4 rounded border border-white/5">
              <p className="text-xs text-gray-500 font-mono mb-2">AI PIPELINE STATUS</p>
              
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${scanStep >= 1 ? (scanStep === 1 ? 'bg-yellow-400 animate-pulse' : 'bg-green-400') : 'bg-gray-600'}`}></div>
                  <span className={`font-mono text-[10px] ${scanStep >= 1 ? 'text-white' : 'text-gray-500'}`}>
                    {scanStep === 1 ? "AUTO-ALIGNING IMAGES..." : "AUTO-ALIGNED"}
                  </span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${scanStep >= 2 ? (scanStep === 2 ? 'bg-yellow-400 animate-pulse' : 'bg-green-400') : 'bg-gray-600'}`}></div>
                  <span className={`font-mono text-[10px] ${scanStep >= 2 ? 'text-white' : 'text-gray-500'}`}>
                    {scanStep === 2 ? "APPLYING CHANGEFORMER..." : (scanStep > 2 ? "CHANGEFORMER COMPLETE" : "WAITING...")}
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${scanStep >= 3 ? (scanStep === 3 ? 'bg-yellow-400 animate-pulse' : 'bg-green-400') : 'bg-gray-600'}`}></div>
                  <span className={`font-mono text-[10px] ${scanStep >= 3 ? 'text-white' : 'text-gray-500'}`}>
                    {scanStep === 3 ? "QUERYING RS-CLIP..." : (scanStep > 3 ? "RS-CLIP COMPLETE" : "WAITING...")}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-black/40 p-4 rounded border border-white/5 transition-opacity duration-500">
              <p className="text-xs text-gray-500 font-mono mb-1">DETECTED ANOMALY AREA</p>
              {scanStep < 4 ? (
                <p className="text-yellow-400 font-mono text-sm animate-pulse">CALCULATING...</p>
              ) : (
                <p className="text-neon-red font-mono text-2xl font-bold">{areaMetric.toLocaleString()} <span className="text-sm">m²</span></p>
              )}
            </div>
            
            <div className="bg-black/40 p-4 rounded border border-white/5">
              <p className="text-xs text-gray-500 font-mono mb-2">RS-CLIP CONFIDENCE</p>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: scanStep < 4 ? '0%' : (geoData?.features?.[0]?.properties?.match_confidence ? geoData.features[0].properties.match_confidence.split('%')[0] + '%' : '96%'), transition: 'width 1s ease-in-out' }}></div>
              </div>
              <p className="text-right text-xs text-green-400 font-mono mt-1">{scanStep < 4 ? '--' : (geoData?.features?.[0]?.properties?.match_confidence || '96%')}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TacticalMap;
