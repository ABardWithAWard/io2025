import React, { createContext, useContext, useState, useEffect } from 'react';
import Cookies from 'js-cookie';
import { initializeApp } from 'firebase/app';
import { getAuth, onAuthStateChanged } from 'firebase/auth';

const AuthContext = createContext(null);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

const firebaseConfig = {
    apiKey: process.env.REACT_APP_PRIVATE_KEY,
    authDomain: `${process.env.REACT_APP_PROJECT_ID}.firebaseapp.com`,
    projectId: process.env.REACT_APP_PROJECT_ID,
    storageBucket: `${process.env.REACT_APP_PROJECT_ID}.appspot.com`,
    messagingSenderId: process.env.REACT_APP_CLIENT_ID,
    appId: process.env.REACT_APP_CLIENT_ID
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export const AuthProvider = ({ children }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [userUid, setUserUid] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let unsubscribe;
        
        try {
            unsubscribe = onAuthStateChanged(auth, 
                (user) => {
                    if (user) {
                        setIsAuthenticated(true);
                        setUserUid(user.uid);
                        fetchCsrfToken();
                    } else {
                        setIsAuthenticated(false);
                        setUserUid(null);
                    }
                    setIsLoading(false);
                },
                (error) => {
                    console.error('Auth state change error:', error);
                    setError(error.message);
                    setIsLoading(false);
                }
            );
        } catch (error) {
            console.error('Error setting up auth state listener:', error);
            setError(error.message);
            setIsLoading(false);
        }

        return () => {
            if (unsubscribe) {
                unsubscribe();
            }
        };
    }, []);

    const fetchCsrfToken = async () => {
        try {
            const response = await fetch('/api/csrf-token/', {
                method: 'GET',
                credentials: 'include'
            });
            if (!response.ok) {
                throw new Error('Failed to fetch CSRF token');
            }
        } catch (error) {
            console.error('Error fetching CSRF token:', error);
            setError(error.message);
        }
    };

    const getCsrfToken = () => {
        return Cookies.get('csrftoken');
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
                await fetchCsrfToken();
            }
            return data.isAuthenticated;
        } catch (error) {
            console.error('Error checking authentication:', error);
            return false;
        }
    };

    const refreshCsrfToken = async () => {
        await fetchCsrfToken();
    };

    const logout = async () => {
        try {
            await auth.signOut();
            setIsAuthenticated(false);
            setUserUid(null);
            Cookies.remove('csrftoken');
        } catch (error) {
            console.error('Error during logout:', error);
            setError(error.message);
        }
    };

    const value = {
        isAuthenticated,
        getCsrfToken,
        logout,
        userUid,
        auth,
        isLoading,
        error,
        checkAuthentication,
        refreshCsrfToken
    };

    if (isLoading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}; 