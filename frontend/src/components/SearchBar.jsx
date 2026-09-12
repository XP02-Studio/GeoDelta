import { useState } from 'react';
import { useAppContext } from '../context/AppContext';

export default function SearchBar() {
  const [query, setQuery] = useState('');
  
  // Safely pull the search function from your Context
  const { initiateSearch, searchError } = useAppContext();

  // Function to run when the button is clicked or Enter is pressed
  const handleSearch = () => {
    if (query.trim() !== '') {
      // Triggers the backend call and 3D globe animation in AppContext
      initiateSearch(query);
    }
  };

  return (
    // 'pointer-events-auto' allows clicking, 'z-50' keeps it on top of the globe
    <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-50 pointer-events-auto bg-gray-900/80 p-2 rounded-lg border border-gray-700 shadow-[0_0_15px_rgba(0,0,0,0.5)]">
      <div className="flex gap-2">
        <input
        type="text"
        className="bg-gray-800 text-teal-400 border border-gray-600 p-2 rounded w-80 focus:outline-none focus:border-teal-500 placeholder-gray-500 font-mono"
        placeholder="Enter target coordinates or name..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()} // Allows hitting 'Enter' to search
        />
      
        <button
        onClick={handleSearch}
        className="bg-gray-800 border border-teal-500 text-teal-500 hover:bg-teal-500 hover:text-gray-900 px-6 py-2 rounded font-bold transition-all tracking-wider"
        >
          SEARCH
        </button>
      </div>
      {searchError && <p className="mt-2 px-1 text-xs font-mono text-red-400">{searchError}</p>}

    </div>
  );
}
