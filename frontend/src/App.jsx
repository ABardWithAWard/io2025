import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NavbarComponent from './components/NavbarComponent.jsx';
import UploadPage from './components/UploadPage';
import ContactPage from './components/ContactPage';
import ProfileComponent from './components/ProfileComponent';
import RedirectToDjangoAdmin from './components/RedirectToDjangoAdmin';
import { AuthProvider } from './AuthContext';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          <NavbarComponent />
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/contact" element={<ContactPage />} />
            <Route path="/profile" element={<ProfileComponent />} />
            <Route path="/admin" element={<RedirectToDjangoAdmin />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App; 