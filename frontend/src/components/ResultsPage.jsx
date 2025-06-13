// Results display component for showing processed document data
// NOTE: Should ONLY be accessed by passing payload which this document processes
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

const ResultsPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { checkAuthentication } = useAuth();
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkAuth = async () => {
            await checkAuthentication();
            setLoading(false);
        };

        checkAuth();
    }, [checkAuthentication]);

    useEffect(() => {
        if (!location.state?.results) {
            // If results are null then go back to upload
            // Should never be called when using app normally
            navigate('/upload');
            return;
        }

        try {
            const parsedResults = JSON.parse(location.state.results);
            setResults(parsedResults);
        } catch (err) {
            setError('Failed to parse results');
            console.error('Error parsing results:', err);
        }
    }, [location.state, navigate]);

    // Check if there is any meaningful data
    // no keys -> undefined (false if used like boolean)
    // only zeroes -> false
    // any other values -> true
    const hasConfidenceValues = results?.confidence?.some(conf => conf > 0);

    if (loading) {
        // Spinner could be added here as well
        return (
            <div className="container mt-4">
                <p>Loading...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mt-4">
                <div className="alert alert-danger" role="alert">
                    {error}
                </div>
            </div>
        );
    }

    if (!results) {
        return (
            <div className="container mt-4">
                <p>No results found</p>
            </div>
        );
    }

    return (
        <div className="container mt-4">
            <h2>OCR Results</h2>
            <div className="row">
                <div className="col-md-6">
                    <div className="card mb-4">
                        <div className="card-header">
                            <h5 className="card-title mb-0">Processed Image</h5>
                        </div>
                        {/* eslint-disable-next-line jsx-a11y/img-redundant-alt */}
                        <img
                            src={`data:image/${results.format};base64,${results.image}`}
                            alt="Processed image"
                            className="card-img-top"
                            style={{ maxHeight: '400px', objectFit: 'contain' }}
                        />
                    </div>
                </div>
                <div className="col-md-6">
                    <div className="card">
                        <div className="card-header">
                            <h5 className="card-title mb-0">Extracted Text</h5>
                        </div>
                        <div className="card-body">
                            {hasConfidenceValues && (
                                <div className="mb-4">
                                    <h6>Confidence Legend:</h6>
                                    <div className="d-flex gap-3 mb-3">
                                        <div className="d-flex align-items-center">
                                            <div className="me-2" style={{ width: '20px', height: '20px', backgroundColor: '#28a745' }}></div>
                                            <span>High (≥90%)</span>
                                        </div>
                                        <div className="d-flex align-items-center">
                                            <div className="me-2" style={{ width: '20px', height: '20px', backgroundColor: '#17a2b8' }}></div>
                                            <span>Good (≥70%)</span>
                                        </div>
                                        <div className="d-flex align-items-center">
                                            <div className="me-2" style={{ width: '20px', height: '20px', backgroundColor: '#ffc107' }}></div>
                                            <span>Fair (≥50%)</span>
                                        </div>
                                        <div className="d-flex align-items-center">
                                            <div className="me-2" style={{ width: '20px', height: '20px', backgroundColor: '#dc3545' }}></div>
                                            <span>Low (&lt;50%)</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div className="mb-3">
                                {results.content.map((text, index) => (
                                    text && (
                                        <span 
                                            key={index}
                                            style={{ 
                                                color: hasConfidenceValues ? getConfidenceColor(results.confidence[index]) : '#000000',
                                                marginRight: '0.5rem'
                                            }}
                                        >
                                            {text}
                                        </span>
                                    )
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Helper function to determine text color based on confidence
const getConfidenceColor = (confidence) => {
    if (confidence >= 0.90) return '#28a745'; // green
    if (confidence >= 0.70) return '#17a2b8'; // blue
    if (confidence >= 0.50) return '#ffc107'; // yellow
    return '#dc3545'; // red
    // Can be adjusted after further testing
};

export default ResultsPage; 