import React, { useState, useEffect } from 'react';
import { useAuth } from '../AuthContext';

const ProfileComponent = () => {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const { isAuthenticated, userUid, checkAuthentication, getCsrfToken, isLoading: authLoading } = useAuth();

    useEffect(() => {
        let isMounted = true;

        const fetchImages = async () => {
            if (!isMounted) return;

            try {
                // Reset states when auth is loading to prevent error on refresh
                if (authLoading) {
                    setLoading(true);
                    setError(null);
                    return;
                }

                const isAuth = await checkAuthentication();
                if (!isAuth || !userUid) {
                    if (isMounted) {
                        setError('Please log in to view your images');
                        setLoading(false);
                        setImages([]);
                    }
                    return;
                }

                const response = await fetch('/api/get-images/', {
                    method: 'POST',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCsrfToken()
                    },
                    body: JSON.stringify({ uid: userUid })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to fetch images');
                }

                const data = await response.json();
                if (isMounted) {
                    setImages(data.images);
                    setLoading(false);
                    setError(null);
                }
            } catch (err) {
                if (isMounted) {
                    setError('Error fetching images: ' + err.message);
                    setLoading(false);
                    setImages([]);
                }
            }
        };

        fetchImages();

        return () => {
            isMounted = false;
        };
    }, [isAuthenticated, userUid, checkAuthentication, getCsrfToken, authLoading]);

    if (authLoading) {
        return <div className="container mt-4"><p>Loading authentication...</p></div>;
    }

    return (
        <div>
            <div className="container mt-4">
                <h1>Your Images</h1>
                {loading && <p>Loading your images...</p>}
                {error && <p className="text-danger">{error}</p>}
                {!loading && !error && (
                    <div className="row">
                        {images.length === 0 ? (
                            <p>No images found</p>
                        ) : (
                            images.map((imageData, index) => (
                                <div key={index} className="col-md-4 mb-4">
                                    <div className="card">
                                        <img 
                                            src={`data:image/png;base64,${imageData}`}
                                            alt={`Uploaded image ${index + 1}`}
                                            className="card-img-top"
                                            style={{ maxHeight: '300px', objectFit: 'contain' }}
                                        />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default ProfileComponent; 