import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

function RedirectToDjangoAdmin() {
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    const checkAdminAccess = async () => {
      try {
        const response = await fetch('/api/auth-status/');
        const data = await response.json();
        
        if (data.isAuthenticated && data.user.is_staff) {
          setIsAdmin(true);
          // Use window.location.href only for the final redirect to Django admin
          // This is necessary because Django admin is a separate application
          window.location.href = '/admin/';
        } else {
          setShowLoginModal(true);
        }
      } catch (error) {
        console.error('Error checking admin access:', error);
        setShowLoginModal(true);
      }
    };

    if (isAuthenticated) {
    checkAdminAccess();
    } else {
      setShowLoginModal(true);
    }
  }, [isAuthenticated]);

  const handleCloseModal = () => {
    setShowLoginModal(false);
    navigate('/');
  };

  return (
    <>
      {showLoginModal && (
        <div className="modal" style={{ display: 'block' }}>
          <div className="modal-content">
            <span className="close" onClick={handleCloseModal}>&times;</span>
            <h2>Brak dostępu</h2>
            <p>Nie masz uprawnień do dostępu do panelu administracyjnego.</p>
            <button onClick={handleCloseModal} className="btn btn-outline-primary">Zamknij</button>
          </div>
        </div>
      )}
    </>
  );
}

export default RedirectToDjangoAdmin; 