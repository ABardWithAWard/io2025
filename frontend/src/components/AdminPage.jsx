// Basic admin page
// A good idea would be to implement clean_firestore.py and set_staff.py here
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {useAuth} from "../AuthContext";

const AdminPage = () => {
    const navigate = useNavigate();
    const [limits, setLimits] = useState({ dataLimit: 0, fileLimit: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const {getCsrfToken, checkAuthentication, isStaff} = useAuth();

    useEffect(() => {
        const checkAdminAccess = async () => {
            try {
                const isAuth = await checkAuthentication();
                if (!isAuth || !isStaff) {
                    navigate('/');
                    return;
                }
                await fetchLimits();
            } catch (error) {
                console.error('Error checking auth status:', error);
                navigate('/');
            }
        };

        checkAdminAccess();
    }, [checkAuthentication, navigate, isStaff]);

    const fetchLimits = async () => {
        // Fetching limits to display them in form
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
        // Submit form to backend, simple logic
        e.preventDefault(); // Prevents undefined behaviour
        // Browsers sometimes handle forms in a different way than we intend to
        // e is short for event
        setError(null);
        setSuccess(null);

        try {
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

    // Documentation is in NavbarComponent, look for function handleInputChange
    const handleChange = (e) => {
        const { name, value } = e.target;
        setLimits(prev => ({
            ...prev,
            [name]: parseInt(value) || 0 // Makes sure we have integer as argument
            // If value is unparsable to integer it returns NaN which makes us submit 0 to form
            // In any other situation it returns parsed value
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