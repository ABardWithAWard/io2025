import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const AdminPage = () => {
    const navigate = useNavigate();

    useEffect(() => {
        const checkAdminAccess = async () => {
            try {
                const response = await fetch('/api/auth-status/', {
                    method: 'GET',
                    credentials: 'include'
                });
                const data = await response.json();
                
                if (data.user.is_staff !== true) {
                    navigate('/');
                }
            } catch (error) {
                console.error('Error checking auth status:', error);
                navigate('/');
            }
        };

        checkAdminAccess();
    }, [navigate]);

    return (
        <div>
            <div className="container mt-4">
                <h1>Admin Page</h1>
                {/* Profile content will be added here */}
            </div>
        </div>
    );
};

export default AdminPage;