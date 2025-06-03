import React, { useState, useEffect } from 'react';
import { useAuth } from '../AuthContext';

const ProfileComponent = () => {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const { isAuthenticated, userUid, checkAuthentication, getCsrfToken } = useAuth();

    useEffect(() => {
        const fetchImages = async () => {
            try {
                const isAuth = await checkAuthentication();
                if (!isAuth || !userUid) {
                    setError('Please log in to view your images');
                    setLoading(false);
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
                setImages(data.images);
                setLoading(false);
            } catch (err) {
                setError('Error fetching images: ' + err.message);
                setLoading(false);
            }
        };

        fetchImages();
    }, [isAuthenticated, userUid, checkAuthentication, getCsrfToken]);

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