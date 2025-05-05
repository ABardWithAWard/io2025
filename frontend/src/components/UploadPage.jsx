import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const UploadPage = () => {
    const [files, setFiles] = useState([]);
    const [showPrivacyDialog, setShowPrivacyDialog] = useState(false);
    const [hasShownPrivacyWarning, setHasShownPrivacyWarning] = useState(false);
    const [error, setError] = useState('');
    const [csrfToken, setCsrfToken] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        // Fetch CSRF token
        fetch('/application/csrf-token/')
            .then(response => response.json())
            .then(data => {
                setCsrfToken(data.csrf_token);
            })
            .catch(error => {
                console.error('Error fetching CSRF token:', error);
            });

        // Fetch files
        fetch('/application/files/')
            .then(response => response.json())
            .then(data => {
                setFiles(data);
            })
            .catch(error => {
                console.error('Error fetching files:', error);
            });
    }, []);

    const validateFile = (file) => {
        const allowedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'];
        const fileName = file.name.toLowerCase();
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    };

    const handleFileUpload = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        const fileInput = document.querySelector('input[type="file"]');
        
        if (fileInput.files.length === 0) {
            setError('Please select a file to upload');
            return;
        }

        formData.append('file', fileInput.files[0]);
        formData.append('csrfmiddlewaretoken', csrfToken);

        try {
            const response = await fetch('/application/upload/', {
                method: 'POST',
                body: formData,
                credentials: 'include',
                headers: {
                    'X-CSRFToken': csrfToken
                }
            });

            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success') {
                    // Refresh the file list
                    const filesResponse = await fetch('/application/files/');
                    const filesData = await filesResponse.json();
                    setFiles(filesData);
                    setError('');
                } else {
                    setError(data.errors || 'Upload failed');
                }
            } else {
                const data = await response.json();
                setError(data.errors || 'Upload failed');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            setError('An error occurred while uploading the file');
        }
    };

    const handlePrivacyContinue = () => {
        setHasShownPrivacyWarning(true);
        setShowPrivacyDialog(false);
        document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
    };

    const handlePrivacyCancel = () => {
        setShowPrivacyDialog(false);
    };

    const handleGoogleLogin = () => {
        window.location.href = '/application/auth/google/';
    };

    const handleLogout = async () => {
        try {
            await fetch('/application/auth/logout/', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'X-CSRFToken': csrfToken
                }
            });
            window.location.href = '/application/';
        } catch (error) {
            console.error('Error logging out:', error);
        }
    };

    return (
        <div className="container mt-4">
            <h2>Upload File</h2>
            <form onSubmit={handleFileUpload} id="uploadForm">
                <div className="mb-3">
                    <label htmlFor="file" className="form-label">Select file to upload:</label>
                    <input type="file" className="form-control" id="file" accept="image/*" />
                </div>
                <button type="submit" className="btn btn-primary">Upload</button>
            </form>
            
            {error && (
                <div className="alert alert-danger mt-3" role="alert">
                    {error}
                </div>
            )}

            <div className="mt-4">
                <h3>Uploaded Files</h3>
                <ul className="list-group">
                    {files.map((file, index) => (
                        <li key={index} className="list-group-item">
                            {file}
                        </li>
                    ))}
                </ul>
            </div>

            <div className="mt-3">
                <button className="btn btn-primary me-2" onClick={handleGoogleLogin}>
                    Login with Google
                </button>
                <button className="btn btn-danger" onClick={handleLogout}>
                    Logout
                </button>
            </div>

            {showPrivacyDialog && (
                <dialog open style={{
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    padding: '20px',
                    borderRadius: '8px',
                    border: '1px solid #ddd',
                    zIndex: 1000
                }}>
                    <h3>Privacy Warning</h3>
                    <p>Please do not upload any private or sensitive information.</p>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                        <button 
                            onClick={handlePrivacyContinue}
                            style={{
                                padding: '8px 16px',
                                backgroundColor: '#0056b3',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Continue
                        </button>
                        <button 
                            onClick={handlePrivacyCancel}
                            style={{
                                padding: '8px 16px',
                                backgroundColor: '#6c757d',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            Cancel
                        </button>
                    </div>
                </dialog>
            )}
        </div>
    );
};

export default UploadPage; 