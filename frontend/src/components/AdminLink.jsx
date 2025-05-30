import React from 'react';
import { Nav } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

const AdminLink = () => {
    const { isAuthenticated, checkAuthentication } = useAuth();
    const [showAccessDenied, setShowAccessDenied] = React.useState(false);
    const [isAdmin, setIsAdmin] = React.useState(false);
    const navigate = useNavigate();

    React.useEffect(() => {
        const checkAdminStatus = async () => {
            const response = await fetch('/api/auth-status/');
            const data = await response.json();
            setIsAdmin(data.isAuthenticated && data.user?.is_staff);
        };
        checkAdminStatus();
    }, [isAuthenticated]);

    const handleClick = (e) => {
        e.preventDefault();
        if (isAdmin) {
            navigate('/admin');
        } else {
            setShowAccessDenied(true);
        }
    };

    const closePopUp = () => {
        setShowAccessDenied(false);
    };

    return (
        <>
            <Nav.Link as={Link} to="/admin" onClick={handleClick}>
                Panel administracji
            </Nav.Link>
            {showAccessDenied && (
                <div className="modal" style={{ display: 'block' }}>
                    <div className="modal-content">
                        <span className="close" onClick={closePopUp}>&times;</span>
                        <h2>Brak dostępu</h2>
                        <p>Nie masz uprawnień do dostępu do panelu administracyjnego.</p>
                        <button onClick={closePopUp} className="btn btn-outline-primary">Zamknij</button>
                    </div>
                </div>
            )}
        </>
    );
};

export default AdminLink; 