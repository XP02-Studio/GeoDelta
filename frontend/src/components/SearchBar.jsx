import { useState, useEffect } from 'react'; //[cite: 1]

export default function SearchBar() { //[cite: 1]
  const [query, setQuery] = useState(''); //[cite: 1]
  const [results, setResults] = useState([]); //[cite: 1]
  const [isSearching, setIsSearching] = useState(false); //[cite: 1]

  // This useEffect watches the 'query' variable in real-time[cite: 1]
  useEffect(() => { //[cite: 1]
    // 1. If the box is empty, clear results and stop.[cite: 1]
    if (query.trim() === '') { //[cite: 1]
      setResults([]); //[cite: 1]
      return; //[cite: 1]
    }

    // Set up an AbortController to cancel out-of-date API requests 
    const controller = new AbortController();
    const signal = controller.signal;

    // 2. The "Debounce" Timer: Wait 300ms after they stop typing[cite: 1]
    const delayDebounceFn = setTimeout(async () => { //[cite: 1]
      setIsSearching(true); //[cite: 1]
      try { //[cite: 1]
        // Safely format the URL to prevent double-slashes causing 404s
        const baseUrl = import.meta.env.VITE_API_URL?.replace(/\/+$/, '') || '';
        
        // Automatically fetch from your live Render URL![cite: 1]
        const response = await fetch(`${baseUrl}/search`, { //[cite: 1]
          method: 'POST', //[cite: 1]
          headers: { 'Content-Type': 'application/json' }, //[cite: 1]
          body: JSON.stringify({  //[cite: 1]
            query_text: query,  //[cite: 1]
            top_k: 5  //[cite: 1]
          }), //[cite: 1]
          signal: signal // Attach the cancellation signal to the fetch request
        }); //[cite: 1]
        
        const data = await response.json(); //[cite: 1]
        setResults(data.results || []); //[cite: 1]
        
      } catch (error) { //[cite: 1]
        // Ignore the error if it was caused by our intentional abort
        if (error.name !== 'AbortError') {
            console.error("Real-time search error:", error); //[cite: 1]
        }
      } finally { //[cite: 1]
        setIsSearching(false); //[cite: 1]
      } //[cite: 1]
    }, 300); // 300ms delay[cite: 1]

    // 3. Cleanup function to cancel the timer if they keep typing[cite: 1]
    return () => {
      clearTimeout(delayDebounceFn); //[cite: 1]
      controller.abort(); // Cancel the pending fetch request if a new keystroke occurs
    };
    
  }, [query]); // <--- This array tells React to run this every time 'query' changes[cite: 1]

  return ( //[cite: 1]
    <div className="search-container">
      <input
        type="text" //[cite: 1]
        placeholder="Search for 'new construction'..." //[cite: 1]
        value={query} //[cite: 1]
        onChange={(e) => setQuery(e.target.value)} // Updates state instantly on typing[cite: 1]
      />
      
      {isSearching && <p>Searching...</p>}
      
      {/* Render your results here */} 
      <ul>
        {results.map((result, index) => ( //[cite: 1]
           <li key={index}>Score: {result.score} - {JSON.stringify(result.metadata)}</li> //[cite: 1]
        ))} 
      </ul>
    </div>
  ); //[cite: 1]
} //[cite: 1]