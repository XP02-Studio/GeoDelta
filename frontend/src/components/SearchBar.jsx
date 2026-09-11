import { useState, useEffect } from 'react';

export default function SearchBar() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // This useEffect watches the 'query' variable in real-time
  useEffect(() => {
    // 1. If the box is empty, clear results and stop.
    if (query.trim() === '') {
      setResults([]);
      return;
    }

    // 2. The "Debounce" Timer: Wait 300ms after they stop typing
    const delayDebounceFn = setTimeout(async () => {
      setIsSearching(true);
      try {
        // Automatically fetch from your live Render URL!
        const response = await fetch(`${import.meta.env.VITE_API_URL}/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            query_text: query, 
            top_k: 5 
          })
        });
        
        const data = await response.json();
        setResults(data.results || []);
        
      } catch (error) {
        console.error("Real-time search error:", error);
      } finally {
        setIsSearching(false);
      }
    }, 300); // 300ms delay

    // 3. Cleanup function to cancel the timer if they keep typing
    return () => clearTimeout(delayDebounceFn);
    
  }, [query]); // <--- This array tells React to run this every time 'query' changes

  return (
    <div className="search-container">
      <input
        type="text"
        placeholder="Search for 'new construction'..."
        value={query}
        onChange={(e) => setQuery(e.target.value)} // Updates state instantly on typing
      />
      
      {isSearching && <p>Searching...</p>}
      
      {/* Render your results here */}
      <ul>
        {results.map((result, index) => (
           <li key={index}>Score: {result.score} - {JSON.stringify(result.metadata)}</li>
        ))}
      </ul>
    </div>
  );
}