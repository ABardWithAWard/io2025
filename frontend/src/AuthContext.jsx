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
    const [idToken, setIdToken] = useState(null);

    // Reset all auth state
    const resetAuthState = () => {
        setIsAuthenticated(false);
        setUserUid(null);
        setIdToken(null);
        setError(null);
        setIsLoading(true);
    };

    useEffect(() => {
        let unsubscribe;
        
        // Reset state on mount
        resetAuthState();
        
        try {
            unsubscribe = onAuthStateChanged(auth, 
                async (user) => {
                    if (user) {
                        try {
                            // Get the ID token
                            const token = await user.getIdToken(true); // Force refresh token
                            setIdToken(token);
                            // Check authentication status with backend
                            const isAuth = await checkAuthentication();
                            if (!isAuth) {
                                resetAuthState();
                            }
                        } catch (error) {
                            console.error('Error getting user token:', error);
                            setError(error.message);
                            resetAuthState();
                        }
                    } else {
                        console.log('User signed out');
                        resetAuthState();
                    }
                    setIsLoading(false);
                },
                (error) => {
                    console.error('Auth state change error:', error);
                    setError(error.message);
                    resetAuthState();
                    setIsLoading(false);
                }
            );
        } catch (error) {
            console.error('Error setting up auth state listener:', error);
            setError(error.message);
            resetAuthState();
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
                credentials: 'include',
                headers: {
                    'Authorization': idToken ? `Bearer ${idToken}` : ''
                }
            });
            const data = await response.json();
            setIsAuthenticated(data.isAuthenticated);
            if (data.isAuthenticated && data.user?.firebase_uid) {
                setUserUid(data.user.firebase_uid);
                await fetchCsrfToken();
            } else {
                setUserUid(null);
            }
            return data.isAuthenticated;
        } catch (error) {
            console.error('Error checking authentication:', error);
            setUserUid(null);
            return false;
        }
    };

    const refreshCsrfToken = async () => {
        await fetchCsrfToken();
    };

    const logout = async () => {
        try {
            await auth.signOut();
            resetAuthState();
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
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100vh',
                backgroundColor: '#f8f9fa'
            }}>
                <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
                    <span className="visually-hidden">Loading...</span>
                </div>
                <p className="mt-3 text-muted">Loading ocr...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100vh',
                backgroundColor: '#f8f9fa'
            }}>
                <div className="alert alert-danger" role="alert">
                    Error: {error}
                </div>
            </div>
        );
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}; 