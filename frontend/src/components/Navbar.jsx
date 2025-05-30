import React, { useState, useEffect } from 'react';
import { useAuth } from '../AuthContext';
import LoginModal from './LoginModal';

function Navbar() {
    const { isAuthenticated, userUid, logout } = useAuth();
    const [showLoginModal, setShowLoginModal] = useState(false);
    const [showAccessDenied, setShowAccessDenied] = useState(false);
    const [isStaff, setIsStaff] = useState(false);

    useEffect(() => {
        const checkStaffStatus = async () => {
            try {
                const response = await fetch('/api/auth-status/');
                const data = await response.json();
                setIsStaff(data.isAuthenticated && data.user?.is_staff);
            } catch (error) {
                console.error('Error checking staff status:', error);
                setIsStaff(false);
            }
        };

        if (isAuthenticated) {
            checkStaffStatus();
        }
    }, [isAuthenticated]);

    const handleAdminClick = (e) => {
        if (!isStaff) {
            e.preventDefault();
            setShowAccessDenied(true);
        }
    };

    const handleLogout = async () => {
        await logout();
        window.location.href = '/';
    };

    return (
        <>
            <nav className="navbar navbar-expand-lg navbar-light bg-light">
                <div className="container-fluid">
                    <a className="navbar-brand" href="/">Aplikacja OCR</a>
                    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span className="navbar-toggler-icon"></span>
                    </button>
                    <div className="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul className="navbar-nav me-auto mb-2 mb-lg-0">
                            <li className="nav-item">
                                <a className="nav-link" href="/contact">Kontakt</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link" href="/admin" onClick={handleAdminClick}>
                                    Panel administracji
                                </a>
                            </li>
                        </ul>
                        <div id="authButtons">
                            {isAuthenticated ? (
                                <>
                                    <span style={{ marginRight: '1rem' }}>{userUid}</span>
                                    <button onClick={handleLogout} className="btn btn-outline-primary">Logout</button>
                                </>
                            ) : (
                                <button onClick={() => setShowLoginModal(true)} className="btn btn-outline-primary">Login</button>
                            )}
                        </div>
                    </div>
                </div>
            </nav>

            {showLoginModal && (
                <LoginModal onClose={() => setShowLoginModal(false)} />
            )}

            {showAccessDenied && (
                <div className="modal" style={{ display: 'block' }}>
                    <div className="modal-content">
                        <span className="close" onClick={() => setShowAccessDenied(false)}>&times;</span>
                        <h2>Brak dostępu</h2>
                        <p>Nie masz uprawnień do dostępu do panelu administracyjnego.</p>
                        <button onClick={() => setShowAccessDenied(false)} className="btn btn-outline-primary">Zamknij</button>
                    </div>
                </div>
            )}
        </>
    );
}

export default Navbar; 