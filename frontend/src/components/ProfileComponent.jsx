// User history page
import React, { useState, useEffect } from 'react';
import { useAuth } from '../AuthContext';
import { Link } from 'react-router-dom';

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

                const isAuth = await checkAuthentication(); // Loads auth from context
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
                    Error(errorData.error || 'Failed to fetch images');
                }

                const data = await response.json();
                if (isMounted) { // If auth is properly mounted display everything
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
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '50vh',
                backgroundColor: '#f8f9fa'
            }}>
                <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
                    <span className="visually-hidden">Loading...</span>
                </div>
                <p className="mt-3 text-muted">Loading profile...</p>
            </div>
        );
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
                            images.map((imageData, index) => {
                                // Format the data to match the UploadPage payload format
                                const resultsPayload = {
                                    name: imageData.filename,
                                    image: imageData.image,
                                    format: imageData.filename.split('.').pop().toLowerCase(),
                                    confidence: imageData.ocr_results.confidence_scores,
                                    content: imageData.ocr_results.text_predictions,
                                    paragraphWidth: imageData.paragraphWidth,
                                    fontSize: imageData.fontSize,
                                };

                                return (
                                    <div key={index} className="col-md-4 mb-4">
                                        <div className="card">
                                            {/* eslint-disable-next-line jsx-a11y/img-redundant-alt */}
                                            <img
                                                src={`data:image/${resultsPayload.format};base64,${resultsPayload.image}`}
                                                alt={`Uploaded image ${index + 1}`}
                                                className="card-img-top"
                                                style={{ maxHeight: '300px', objectFit: 'contain' }}
                                            />
                                            <div className="card-body">
                                                <h5 className="card-title">{resultsPayload.name}</h5>
                                                <p className="card-text">
                                                    {resultsPayload.content?.slice(0, 3).join(' ')}...
                                                </p>
                                                <Link 
                                                    to="/results"
                                                    state={{ results: JSON.stringify(resultsPayload) }}
                                                    className="btn btn-primary"
                                                >
                                                    View Full Results
                                                </Link>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default ProfileComponent; 