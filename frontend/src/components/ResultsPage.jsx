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

    if (loading) {
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
                            <div className="mb-3">
                                {results.content.map((text, index) => (
                                    text && (
                                        <div key={index} className="mb-3 p-2 border-bottom">
                                            <div className="d-flex justify-content-between align-items-center">
                                                <span className="fs-5">{text}</span>
                                                {results.confidence[index] > 0 && (
                                                    <span className={`badge ${getConfidenceBadgeColor(results.confidence[index])}`}>
                                                        Confidence: {results.confidence[index]}%
                                                    </span>
                                                )}
                                            </div>
                                        </div>
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

// Helper function to determine badge color based on confidence
// Can be adjusted accordingly
const getConfidenceBadgeColor = (confidence) => {
    if (confidence >= 0.90) return 'bg-success';
    if (confidence >= 0.70) return 'bg-info';
    if (confidence >= 0.50) return 'bg-warning';
    return 'bg-danger';
};

export default ResultsPage; 