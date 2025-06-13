// Main application component that sets up routing, app display and authentication
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavbarComponent from './components/NavbarComponent.jsx';
import UploadPage from './components/UploadPage';
import ContactPage from './components/ContactPage';
import ProfileComponent from './components/ProfileComponent';
import AdminPage from './components/AdminPage';
import ResultsPage from './components/ResultsPage';
import { AuthProvider } from './AuthContext';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          {/* Makes navbar appear in all links supported by routes */}
          <NavbarComponent />
          <Routes>
            {/* Main routes for the application, just like urls file in django */}
            <Route path="/" element={<UploadPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/contact" element={<ContactPage />} />
            <Route path="/profile" element={<ProfileComponent />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App; 