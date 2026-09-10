import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Platform from './pages/Platform';

const API_BASE_URL = import.meta.env.VITE_API_URL;

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/app" element={<Platform />} />
    </Routes>
  );
}

export default App;
