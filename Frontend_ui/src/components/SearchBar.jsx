import React, { useState } from 'react';
import { Search, MapPin } from 'lucide-react';
import { useAppContext } from '../context/AppContext';

const SearchBar = () => {
  const { initiateSearch, isAnimatingToTarget, is2DView } = useAppContext();
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isAnimatingToTarget) {
      initiateSearch(input.trim());
      setInput('');
    }
  };

  // Hide search bar when in 2D view to let map take over, or keep it.
  // The requirement says "Tactical Header... entry point for your NLP semantic search".
  // It looks cooler if it stays at the top.
  
  return (
    <div className={`absolute top-6 left-1/2 transform -translate-x-1/2 z-40 pointer-events-auto transition-opacity duration-500 ${is2DView ? 'opacity-0' : 'opacity-100'}`}>
      <form onSubmit={handleSubmit} className="relative group">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className={`w-5 h-5 ${isAnimatingToTarget ? 'text-neon-red' : 'text-neon-cyan'} group-focus-within:glow-cyan`} />
        </div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isAnimatingToTarget ? "CALCULATING TRAJECTORY..." : "Search 'new runway' or 'cleared forest'..."}
          disabled={isAnimatingToTarget}
          className="bg-tactical-dark/60 border border-white/20 text-white text-sm font-mono rounded-full focus:outline-none focus:border-neon-cyan focus:bg-tactical-dark/80 block w-96 pl-10 p-2.5 backdrop-blur-md transition-all shadow-[0_0_15px_rgba(0,243,255,0.1)] focus:shadow-[0_0_20px_rgba(0,243,255,0.4)] disabled:opacity-50"
        />
        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
          <MapPin className="w-4 h-4 text-gray-500" />
        </div>
      </form>
    </div>
  );
};

export default SearchBar;
