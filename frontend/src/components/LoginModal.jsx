import React, { useState } from 'react';

const LoginModal = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState('login');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Implement login/register logic
  };

  if (!isOpen) return null;

  return (
    <div className="modal" style={{ display: 'block' }}>
      <div className="modal-content">
        <span className="close" onClick={onClose}>&times;</span>
        
        <div className="tab">
          <button 
            className={`tablinks ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => setActiveTab('login')}
          >
            Login
          </button>
          <button 
            className={`tablinks ${activeTab === 'register' ? 'active' : ''}`}
            onClick={() => setActiveTab('register')}
          >
            Register
          </button>
        </div>

        <div id="loginTab" className="tabcontent" style={{ display: activeTab === 'login' ? 'block' : 'none' }}>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="loginEmail">Email</label>
              <input type="email" id="loginEmail" name="email" required />
            </div>
            <div className="form-group">
              <label htmlFor="loginPassword">Password</label>
              <input type="password" id="loginPassword" name="password" required />
            </div>
            <button type="submit" className="submit-btn">Login</button>
            <a href="/api/auth/google" className="google-btn" style={{ width: '100%', justifyContent: 'center', marginLeft: 0 }}>
              <img src="https://www.google.com/favicon.ico" alt="Google" width="18" />
              Login with Google
            </a>
          </form>
        </div>

        <div id="registerTab" className="tabcontent" style={{ display: activeTab === 'register' ? 'block' : 'none' }}>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="registerEmail">Email</label>
              <input type="email" id="registerEmail" name="email" required />
            </div>
            <div className="form-group">
              <label htmlFor="registerPassword">Password</label>
              <input type="password" id="registerPassword" name="password" required />
            </div>
            <div className="form-group">
              <label htmlFor="registerConfirmPassword">Confirm Password</label>
              <input type="password" id="registerConfirmPassword" name="confirm_password" required />
            </div>
            <button type="submit" className="submit-btn">Register</button>
            <a href="/api/auth/google" className="google-btn" style={{ width: '100%', justifyContent: 'center', marginLeft: 0 }}>
              <img src="https://www.google.com/favicon.ico" alt="Google" width="18" />
              Register with Google
            </a>
          </form>
        </div>
      </div>
    </div>
  );
};

export default LoginModal; 