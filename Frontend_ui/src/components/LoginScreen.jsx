import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { Shield, Fingerprint, Lock } from 'lucide-react';

const LoginScreen = () => {
  const { setIsAuthenticated } = useAppContext();
  const [isFading, setIsFading] = useState(false);

  const handleGuestLogin = () => {
    // Optional audio cue
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(150, audioCtx.currentTime); // Low freq
      oscillator.frequency.exponentialRampToValueAtTime(40, audioCtx.currentTime + 0.5);
      gainNode.gain.setValueAtTime(0.5, audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      oscillator.start();
      oscillator.stop(audioCtx.currentTime + 0.5);
    } catch (e) {
      console.warn("Audio not supported or blocked", e);
    }

    setIsFading(true);
    // Wait for fade out animation before triggering context state
    setTimeout(() => {
      setIsAuthenticated(true);
    }, 800);
  };

  return (
    <div className={`w-full h-full flex items-center justify-center bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center transition-all duration-800 ${isFading ? 'opacity-0 scale-105 blur-md' : 'opacity-100 scale-100 blur-0'}`}>
      
      {/* Dark Overlay */}
      <div className="absolute inset-0 bg-tactical-dark/70 backdrop-blur-sm"></div>

      <div className="glass-panel w-full max-w-md p-8 relative z-10 flex flex-col items-center">
        <div className="w-16 h-16 rounded-full bg-neon-cyan/20 flex items-center justify-center mb-6 glow-cyan">
          <Shield className="w-8 h-8 text-neon-cyan" />
        </div>
        
        <h1 className="text-3xl font-mono font-bold tracking-widest text-white mb-2">GEODELTA</h1>
        <p className="text-gray-400 font-mono text-xs tracking-widest mb-8 text-center px-4">OFFLINE TACTICAL AI SATELLITE ANALYSIS</p>

        <div className="w-full space-y-4">
          <div className="relative">
            <Lock className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
            <input 
              type="password" 
              placeholder="ENTER CLEARANCE CODE" 
              disabled
              className="w-full bg-black/40 border border-white/10 rounded px-10 py-3 text-white font-mono text-sm focus:outline-none focus:border-neon-cyan transition-colors"
            />
          </div>

          <button 
            className="w-full bg-white/5 hover:bg-white/10 border border-white/20 text-white font-mono py-3 px-4 rounded transition-all cursor-not-allowed text-sm opacity-50"
            disabled
          >
            AUTHORIZE
          </button>

          <div className="flex items-center justify-center space-x-4 py-4">
            <div className="h-px bg-white/20 w-full"></div>
            <span className="text-xs font-mono text-gray-500">OR</span>
            <div className="h-px bg-white/20 w-full"></div>
          </div>

          <button 
            onClick={handleGuestLogin}
            className="w-full bg-neon-cyan/10 hover:bg-neon-cyan/20 border border-neon-cyan text-neon-cyan font-mono py-3 px-4 rounded transition-all flex items-center justify-center group glow-cyan"
          >
            <Fingerprint className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
            GUEST MODE (JUDGE ACCESS)
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;
