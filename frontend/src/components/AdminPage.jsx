import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Cookies from 'js-cookie';

const AdminPage = () => {
    const navigate = useNavigate();
    const [limits, setLimits] = useState({ dataLimit: 0, fileLimit: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);

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
                } else {
                    // If admin, fetch the limits
                    await fetchCsrfToken();
                    fetchLimits();
                }
            } catch (error) {
                console.error('Error checking auth status:', error);
                navigate('/');
            }
        };

        checkAdminAccess();
    }, [navigate]);

    const fetchLimits = async () => {
        try {
            const response = await fetch('/api/global-settings/', {
                method: 'GET',
                credentials: 'include'
            });
            const data = await response.json();
            setLimits(data);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching limits:', error);
            setError('Failed to fetch limits');
            setLoading(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setSuccess(null);

        try {
            await fetchCsrfToken();
            const response = await fetch('/api/global-settings/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                credentials: 'include',
                body: JSON.stringify(limits)
            });

            const data = await response.json();
            
            if (response.ok) {
                setSuccess('Limits updated successfully');
                setLimits(data);
            } else {
                setError(data.error || 'Failed to update limits');
            }
        } catch (error) {
            console.error('Error updating limits:', error);
            setError('Failed to update limits');
        }
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setLimits(prev => ({
            ...prev,
            [name]: parseInt(value) || 0
        }));
    };

    if (loading) {
        return <div className="container mt-4">Loading...</div>;
    }

    return (
        <div>
            <div className="container mt-4">
                <h1>Admin Page</h1>
                
                <div className="card mt-4">
                    <div className="card-header">
                        <h2>Global Settings</h2>
                    </div>
                    <div className="card-body">
                        {error && (
                            <div className="alert alert-danger" role="alert">
                                {error}
                            </div>
                        )}
                        {success && (
                            <div className="alert alert-success" role="alert">
                                {success}
                            </div>
                        )}
                        
                        <form onSubmit={handleSubmit}>
                            <div className="mb-3">
                                <label htmlFor="dataLimit" className="form-label">Data Limit (MB)</label>
                                <input
                                    type="number"
                                    className="form-control"
                                    id="dataLimit"
                                    name="dataLimit"
                                    value={limits.dataLimit}
                                    onChange={handleChange}
                                    min="0"
                                    required
                                />
                            </div>
                            
                            <div className="mb-3">
                                <label htmlFor="fileLimit" className="form-label">File Limit (count)</label>
                                <input
                                    type="number"
                                    className="form-control"
                                    id="fileLimit"
                                    name="fileLimit"
                                    value={limits.fileLimit}
                                    onChange={handleChange}
                                    min="0"
                                    required
                                />
                            </div>
                            
                            <button type="submit" className="btn btn-primary">
                                Update Limits
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdminPage;