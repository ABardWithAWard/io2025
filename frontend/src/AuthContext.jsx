import React, { createContext, useContext, useState, useEffect } from 'react';
import Cookies from 'js-cookie';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [csrfToken, setCsrfToken] = useState(null);
    const [error, setError] = useState('');

    const refreshCsrfToken = async () => {
        try {
            const response = await fetch('/api/csrf-token/', {
                method: 'GET',
                credentials: 'include'
            });
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                const data = await response.json();
                setCsrfToken(data.csrf_token);
                return data.csrf_token;
            } else {
                const text = await response.text();
                console.error('Non-JSON response for CSRF token:', text);
                setError('Unexpected response fetching CSRF token');
                return null;
            }
        } catch (error) {
            console.error('Error fetching CSRF token:', error);
            setError('An error occurred fetching CSRF token');
            return null;
        }
    };

    useEffect(() => {
        // Get CSRF token from cookies on initial load
        refreshCsrfToken();
    }, []);

    const getCsrfToken = () => {
        return csrfToken;
    };

    const checkAuthentication = async () => {
        try {
            const response = await fetch('/api/auth-status/', {
                method: 'GET',
                credentials: 'include'
            });
            const data = await response.json();
            setIsAuthenticated(data.isAuthenticated);
            if (data.isAuthenticated) {
                // Refresh CSRF token when authentication state changes
                await refreshCsrfToken();
            }
            return data.isAuthenticated;
        } catch (error) {
            console.error('Error checking authentication:', error);
            return false;
        }
    };

    const logout = async () => {
        try {
            const response = await fetch('/api/logout/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                credentials: 'include'
            });
            
            if (response.ok) {
                Cookies.remove('csrftoken');
                setIsAuthenticated(false);
                setCsrfToken(null);
                // Refresh CSRF token after logout
                await refreshCsrfToken();
            }
        } catch (error) {
            console.error('Error during logout:', error);
        }
    };

    const value = {
        isAuthenticated,
        csrfToken,
        getCsrfToken,
        checkAuthentication,
        logout,
        error,
        refreshCsrfToken
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export default AuthContext; 