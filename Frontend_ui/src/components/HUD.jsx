import React from 'react';
import { Activity, Database, Cpu, Wifi, LogOut } from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';

const HUD = () => {
  const { setIsAuthenticated, is2DView } = useAppContext();
  const navigate = useNavigate();

  const handleLogout = () => {
    setIsAuthenticated(false);
    navigate('/');
  };

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between overflow-hidden">
      
      {/* Top Left - Project Info */}
      <div className={`absolute top-6 transition-all duration-500 ${is2DView ? 'left-[344px]' : 'left-6'} flex flex-col space-y-2`}>
        <div className="flex items-center space-x-2 text-neon-cyan bg-tactical-dark/50 backdrop-blur-md px-3 py-1.5 border border-neon-cyan/30 rounded w-max">
          <Activity className="w-4 h-4 animate-pulse" />
          <span className="font-mono text-xs font-bold tracking-widest">GEODELTA</span>
        </div>
        <div className="flex items-center space-x-2 text-green-400 bg-tactical-dark/50 backdrop-blur-md px-3 py-1.5 border border-green-400/30 rounded w-max">
          <Wifi className="w-4 h-4" />
          <span className="font-mono text-xs tracking-wider">AIR-GAPPED [SECURE]</span>
        </div>
      </div>

      {/* Top Right - Sign Out */}
      <div className={`absolute top-6 transition-all duration-500 ${is2DView ? 'right-[344px]' : 'right-6'} pointer-events-auto`}>
        <button 
          onClick={handleLogout}
          className="flex items-center space-x-2 bg-red-900/40 hover:bg-red-900/60 border border-red-500/50 text-red-400 px-3 py-1.5 rounded transition-all backdrop-blur-md"
        >
          <LogOut className="w-4 h-4" />
          <span className="font-mono text-xs tracking-widest">SIGN OUT</span>
        </button>
      </div>

      {/* Bottom Right - Telemetry */}
      <div className={`absolute bottom-6 transition-all duration-500 ${is2DView ? 'right-[344px]' : 'right-6'} flex flex-col space-y-2 text-right items-end`}>
        <div className="flex items-center space-x-2 text-gray-300 bg-tactical-dark/50 backdrop-blur-md px-3 py-1.5 border border-white/10 rounded w-max">
          <span className="font-mono text-xs tracking-wider text-gray-400">LOCAL VECTOR DB:</span>
          <span className="font-mono text-xs text-green-400">CONNECTED</span>
          <Database className="w-4 h-4 text-green-400 ml-2" />
        </div>
        <div className="flex items-center space-x-2 text-gray-300 bg-tactical-dark/50 backdrop-blur-md px-3 py-1.5 border border-white/10 rounded w-max">
          <span className="font-mono text-xs tracking-wider text-gray-400">ACTIVE AI CORE:</span>
          <span className="font-mono text-xs text-white">ChangeFormer + RS-CLIP</span>
          <Cpu className="w-4 h-4 text-neon-cyan ml-2" />
        </div>
        <div className="flex items-center space-x-2 text-gray-300 bg-tactical-dark/50 backdrop-blur-md px-3 py-1.5 border border-white/10 rounded w-max">
          <span className="font-mono text-xs tracking-wider text-gray-400">INFERENCE LATENCY:</span>
          <span className="font-mono text-xs text-neon-red font-bold">~420ms</span>
          <Activity className="w-4 h-4 text-neon-red ml-2" />
        </div>
      </div>

      {/* Crosshairs Overlay (subtle) */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-20 pointer-events-none">
        <div className="w-px h-16 bg-neon-cyan absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>
        <div className="w-16 h-px bg-neon-cyan absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>
        <div className="w-4 h-4 border border-neon-cyan rounded-full absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>
      </div>

    </div>
  );
};

export default HUD;
