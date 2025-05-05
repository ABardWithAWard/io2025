// src/components/AppLayout.jsx
import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import NavbarComponent from './NavbarComponent.jsx';

const AppLayout = ({ children }) => {
  return (
    <div>
      <NavbarComponent/>
      <main className="container mt-4">
        {children}
      </main>
    </div>
  );
};

export default AppLayout;