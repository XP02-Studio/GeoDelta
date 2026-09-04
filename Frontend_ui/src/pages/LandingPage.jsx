import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, Clock, CloudOff, Crosshair, Layers, SearchCode } from 'lucide-react';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="w-full min-h-screen bg-tactical-dark text-white font-sans overflow-x-hidden selection:bg-neon-cyan selection:text-black">
      
      {/* Navigation */}
      <nav className="fixed w-full z-50 px-8 py-6 flex justify-between items-center bg-tactical-dark/80 backdrop-blur-md border-b border-white/5">
        <div className="font-mono text-xl font-bold tracking-widest text-white flex items-center gap-2">
          <div className="w-6 h-6 rounded bg-neon-cyan/20 border border-neon-cyan flex items-center justify-center glow-cyan">
            <span className="text-neon-cyan text-xs">Δ</span>
          </div>
          GEODELTA
        </div>
        <button 
          onClick={() => navigate('/app')}
          className="bg-neon-cyan/10 hover:bg-neon-cyan/20 border border-neon-cyan text-neon-cyan font-mono py-2 px-6 rounded transition-all glow-cyan text-sm tracking-widest"
        >
          LAUNCH PLATFORM
        </button>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-40 pb-20 px-8 flex flex-col items-center justify-center min-h-[90vh] bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center">
        <div className="absolute inset-0 bg-tactical-dark/80 backdrop-blur-[2px]"></div>
        
        <div className="relative z-10 max-w-4xl mx-auto text-center flex flex-col items-center">
          <div className="inline-block border border-neon-cyan/50 text-neon-cyan bg-neon-cyan/5 px-4 py-1.5 rounded-full font-mono text-xs tracking-widest mb-8 uppercase">
            Offline, AI-Powered Satellite Analysis
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-8 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
            Spot Tactical Changes.<br/>In Plain English.
          </h1>
          <p className="text-gray-400 text-lg md:text-xl max-w-2xl mb-12 font-light leading-relaxed">
            GeoDelta instantly spots tactical changes on the ground—like newly built bunkers, airstrips, or roads—and lets defense analysts search satellite maps without needing the internet.
          </p>
          <button 
            onClick={() => navigate('/app')}
            className="group relative inline-flex items-center justify-center px-8 py-4 font-mono font-bold text-black transition-all duration-200 bg-neon-cyan rounded-lg hover:bg-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-neon-cyan glow-cyan"
          >
            INITIALIZE GEODELTA
            <Crosshair className="ml-3 w-5 h-5 group-hover:rotate-90 transition-transform duration-300" />
          </button>
        </div>
      </section>

      {/* The Problem Section */}
      <section className="py-24 px-8 border-t border-white/5 bg-[#0a0a0f]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-neon-red font-mono text-sm tracking-widest mb-4">THE PROBLEM</h2>
            <h3 className="text-3xl font-bold text-white">Why Traditional Tools Fail</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="glass-panel p-8 hover:border-neon-red/50 transition-colors">
              <ShieldAlert className="w-10 h-10 text-neon-red mb-6" />
              <h4 className="text-xl font-bold mb-3">Too Many False Alarms</h4>
              <p className="text-gray-400 text-sm leading-relaxed">Traditional image comparison tools trigger alarms simply because clouds moved, seasons changed, or the sun created new shadows.</p>
            </div>
            
            <div className="glass-panel p-8 hover:border-yellow-400/50 transition-colors">
              <Clock className="w-10 h-10 text-yellow-400 mb-6" />
              <h4 className="text-xl font-bold mb-3">Slow Manual Work</h4>
              <p className="text-gray-400 text-sm leading-relaxed">Military personnel still spend countless hours manually hunting through gigabytes of satellite images to find critical differences.</p>
            </div>

            <div className="glass-panel p-8 hover:border-neon-cyan/50 transition-colors">
              <CloudOff className="w-10 h-10 text-neon-cyan mb-6" />
              <h4 className="text-xl font-bold mb-3">Zero Cloud Access</h4>
              <p className="text-gray-400 text-sm leading-relaxed">Defense installations cannot upload sensitive border satellite imagery to cloud servers like AWS or Google due to strict air-gapped security.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="py-24 px-8 border-t border-white/5 bg-tactical-dark">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-neon-cyan font-mono text-sm tracking-widest mb-4">THE SOLUTION</h2>
            <h3 className="text-3xl font-bold text-white">How GeoDelta Works in 3 Steps</h3>
          </div>

          <div className="space-y-12">
            {/* Step 1 */}
            <div className="flex flex-col md:flex-row items-center gap-12 bg-black/40 p-8 rounded-2xl border border-white/5">
              <div className="w-16 h-16 rounded-full bg-white/5 border border-white/20 flex flex-shrink-0 items-center justify-center font-mono text-2xl text-neon-cyan">1</div>
              <div className="flex-grow">
                <h4 className="text-2xl font-bold mb-2 flex items-center gap-3">
                  <Layers className="w-6 h-6 text-neon-cyan" /> Auto-Align
                </h4>
                <p className="text-gray-400">The system takes two satellite images of the same area taken at different times and perfectly aligns them to fix camera angles.</p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col md:flex-row items-center gap-12 bg-black/40 p-8 rounded-2xl border border-white/5">
              <div className="w-16 h-16 rounded-full bg-white/5 border border-white/20 flex flex-shrink-0 items-center justify-center font-mono text-2xl text-neon-cyan">2</div>
              <div className="flex-grow">
                <h4 className="text-2xl font-bold mb-2 flex items-center gap-3">
                  <Crosshair className="w-6 h-6 text-neon-cyan" /> Smart Change Detection
                </h4>
                <p className="text-gray-400">An offline AI model <span className="text-white font-mono text-xs bg-white/10 px-2 py-1 rounded mx-1">ChangeFormer</span> ignores weather, clouds, and seasonal lighting to highlight only real, physical structural changes.</p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col md:flex-row items-center gap-12 bg-black/40 p-8 rounded-2xl border border-white/5">
              <div className="w-16 h-16 rounded-full bg-white/5 border border-white/20 flex flex-shrink-0 items-center justify-center font-mono text-2xl text-neon-cyan">3</div>
              <div className="flex-grow">
                <h4 className="text-2xl font-bold mb-2 flex items-center gap-3">
                  <SearchCode className="w-6 h-6 text-neon-cyan" /> Plain-English Search
                </h4>
                <p className="text-gray-400">Instead of writing code, an operator simply types <span className="text-neon-cyan italic">"new runway"</span> or <span className="text-neon-cyan italic">"cleared forest"</span>. The system uses custom satellite AI <span className="text-white font-mono text-xs bg-white/10 px-2 py-1 rounded mx-1">RS-CLIP</span> and a local vector search engine to pinpoint the exact coordinates.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center border-t border-white/5 text-gray-600 font-mono text-xs">
        &copy; {new Date().getFullYear()} GeoDelta (SIH 2026). All systems operational.
      </footer>
    </div>
  );
};

export default LandingPage;
