import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavbarComponent from './components/NavbarComponent.jsx';
import UploadPage from './components/UploadPage';
import ContactPage from './components/ContactPage';
import ProfileComponent from './components/ProfileComponent';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <Router>
      <div className="App">
        <NavbarComponent />
        <Routes>
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/profile" element={<ProfileComponent />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 