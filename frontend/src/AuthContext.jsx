// Authentication context provider for managing user authentication state
import React, { createContext, useContext, useState, useEffect } from 'react';
import Cookies from 'js-cookie';
import { initializeApp } from 'firebase/app';
import { getAuth, onAuthStateChanged } from 'firebase/auth';

// Create authentication context on start
const AuthContext = createContext(null);

// Custom hook for accessing auth context, this should be used in components
// instead of calling any authentication endpoints
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

// Firebase configuration, should be changed if app ever makes it to prod
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
    const [userEmail, setUserEmail] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [idToken, setIdToken] = useState(null);

    // Reset all auth state
    const resetAuthState = () => {
        setIsAuthenticated(false);
        setUserUid(null);
        setUserEmail(null);
        setIdToken(null);
        setError(null);
        setIsLoading(true);
    };

    useEffect(() => {
        let unsubscribe;
        
        // Reset state on mount
        resetAuthState();
        
        // Unified error handling for authentication flow, used to be a mess
        // This try-catch block handles:
        // 1. Firebase auth state changes
        // 2. Token retrieval and validation
        // 3. Backend authentication status check
        try {
            unsubscribe = onAuthStateChanged(auth, 
                async (user) => {
                    if (user) {
                        // Get the ID token and validate with backend
                        const token = await user.getIdToken(true);
                        setIdToken(token);
                        const isAuth = await checkAuthentication();
                        if (!isAuth) {
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
            console.error('Authentication error:', error);
            setError(error.message);
            resetAuthState();
            setIsLoading(false);
        }

        return () => {
            if (unsubscribe) {
                unsubscribe(); // Makes sure we reset all states on mount
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
                Error('Failed to fetch CSRF token');
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
            // If user is authenticated and firebase_uid is not a null, then set it for
            // image saving purposes
            if (data.isAuthenticated && data.user?.firebase_uid) {
                setUserUid(data.user.firebase_uid);
                setUserEmail(data.user.email);
                await fetchCsrfToken();
            } else {
                setUserUid(null);
                setUserEmail(null);
                // If user is not authenticated set it to null to prevent saving images
                // to somebody's account
            }
            return data.isAuthenticated;
        } catch (error) {
            console.error('Error checking authentication:', error);
            setUserUid(null);
            setUserEmail(null);
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
        userEmail,
        auth,
        isLoading,
        error,
        checkAuthentication,
        refreshCsrfToken
    }; // All values which can be used by other components

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
                {/* Good-looking spinner when loading context instead of boring text*/}
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
                {/* Debug screen if auth context fails to load for any reason.
            It should never be displayed normally (unless local errors like network error are present )*/}
            </div>
        );
    }

    return (
        <AuthContext.Provider value={value}>
            {/* Look at App.jsx to see how AuthContext is implemented there,
              if you do not understand why this return looks like this */}
            {children}
        </AuthContext.Provider>
    );
}; 