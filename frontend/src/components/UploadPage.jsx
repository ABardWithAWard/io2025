import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import data from "bootstrap/js/src/dom/data";

const UploadPage = () => {
    const [files, setFiles] = useState([]);
    const [showPrivacyDialog, setShowPrivacyDialog] = useState(false);
    const [hasShownPrivacyWarning, setHasShownPrivacyWarning] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const [pendingFile, setPendingFile] = useState(null);
    const [showUploadModal, setShowUploadModal] = useState(false);
    const { getCsrfToken } = useAuth();

    useEffect(() => {
        // Fetch files
        fetchFiles()
    }, []);

    const fetchFiles = () => {
        fetch('/api/files/list_files', {
            method: 'GET',
            credentials: 'include'
        })
            .then(async response => {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const data = await response.json();
                    setFiles(data);
                } else {
                    const text = await response.text();
                    console.error('Non-JSON response for files:', text);
                    setError('Unexpected response fetching files');
                }
            })
            .catch(error => {
                console.error('Error fetching files:', error);
                setError('An error occurred fetching files');
            });
    };

    const validateFile = (file) => {
        const allowedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'];
        const fileName = file.name.toLowerCase();
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    };

    const handleUploadClick = (event) => {
        event.preventDefault();
        const fileInput = document.querySelector('input[type="file"]');
        if (!fileInput || fileInput.files.length === 0) {
            setError('Please select a file to upload');
            return;
        }
        setShowUploadModal(true);
    };

    const handlePrivacyContinue = () => {
        setHasShownPrivacyWarning(true);
        setShowPrivacyDialog(false);
        document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
    };

    const handlePrivacyCancel = () => {
        setShowPrivacyDialog(false);
    };

    const handleCancelUpload = () => {
        setShowUploadModal(false);
    };

    const handleConfirmUpload = async () => {
        setShowUploadModal(false);
        await handleFileUpload();
    };

    const handleFileUpload = async () => {
        const formData = new FormData();
        const fileInput = document.querySelector('input[type="file"]');

        if (fileInput.files.length === 0) {
            setError('Please select a file to upload');
            return;
        }

        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/api/files/upload/', {
                method: 'POST',
                body: formData,
                credentials: 'include',
                headers: {
                    'X-CSRFToken': getCsrfToken()
                }
            });

            const data = await response.json();
            if (response.ok) {
                fetchFiles();
                setError('');
            } else {
                setError(data.errors || 'Upload failed');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            setError('An error occurred while uploading the file');
        }
    };

    return (
        <div className="container mt-4">
            <h2>Upload File</h2>
            <form onSubmit={handleUploadClick} id="uploadForm">
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
                            <strong>File:</strong> {file} <br />
                        </li>
                    ))}
                </ul>
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

            {showUploadModal && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    width: '100vw',
                    height: '100vh',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)', // dim effect
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 1000,
                }}>
                    <div style={{
                        backgroundColor: '#fff',
                        padding: '30px',
                        borderRadius: '10px',
                        width: '90%',
                        maxWidth: '400px',
                        boxShadow: '0 0 15px rgba(0,0,0,0.3)',
                        textAlign: 'center',
                    }}>
                        <h3>Confirm Upload</h3>
                        <p>Are you sure you want to upload this file?</p>
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
                            <button
                                onClick={handleConfirmUpload}
                                className="btn btn-success"
                            >
                                Continue
                            </button>
                            <button
                                onClick={handleCancelUpload}
                                className="btn btn-secondary"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
};

export default UploadPage;
