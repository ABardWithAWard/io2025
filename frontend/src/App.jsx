import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavbarComponent from './components/NavbarComponent.jsx';
import UploadPage from './components/UploadPage';
import ContactPage from './components/ContactPage';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <Router>
      <div className="App">
        <NavbarComponent />
        <Routes>
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/" element={<UploadPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 