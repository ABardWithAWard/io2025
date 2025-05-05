import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import UploadPage from './components/UploadPage';
import NavbarComponent from './components/NavbarComponent.jsx';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <NavbarComponent />
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/application" element={<UploadPage />} />
          <Route path="/application/upload" element={<UploadPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
