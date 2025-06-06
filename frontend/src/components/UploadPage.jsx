import React, { useState } from 'react';
import { useAuth } from '../AuthContext';
import { useNavigate } from 'react-router-dom';

const UploadPage = () => {
    const [showPrivacyDialog, setShowPrivacyDialog] = useState(false);
    const [setHasShownPrivacyWarning] = useState(false);
    const [error, setError] = useState('');
    const [showUploadModal, setShowUploadModal] = useState(false);
    const {getCsrfToken} = useAuth();
    const [fontSize, setFontSize] = useState(12);
    const [language, setLanguage] = useState('english');
    const [exportFormat, setExportFormat] = useState('docx');
    const [hasConfidence, setHasConfidence] = useState(false);
    const [paragraphWidth, setParagraphWidth] = useState(80);
    const navigate = useNavigate();

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

        try {
            const authResponse = await fetch('/api/auth-status/', {
                method: 'GET',
                credentials: 'include'
            });
            const authData = await authResponse.json();

            formData.append('file', fileInput.files[0]);
            formData.append('userUid', authData.isAuthenticated ? authData.user.firebase_uid : null);
            formData.append('fontSize', fontSize);
            formData.append('language', language);
            formData.append('format', exportFormat);
            formData.append('confidence', hasConfidence);
            formData.append('paragraphWidth', paragraphWidth);

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
                setError('');
                // Navigate to results page with the OCR results from payload
                navigate('/results', { state: { results: data.payload } });
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

                <div className="row mb-3">
                    <div className="col-md-6">
                        <label htmlFor="language" className="form-label">Language:</label>
                        <select 
                            className="form-select" 
                            id="language"
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                        >
                            <option value="english">English</option>
                            <option value="polish">Polish</option>
                        </select>
                    </div>
                    <div className="col-md-6">
                        <label htmlFor="exportFormat" className="form-label">Export Format:</label>
                        <select 
                            className="form-select" 
                            id="exportFormat"
                            value={exportFormat}
                            onChange={(e) => setExportFormat(e.target.value)}
                        >
                            <option value="docx">DOCX</option>
                            <option value="txt">TXT</option>
                        </select>
                    </div>
                </div>

                <div className="row mb-3">
                    <div className="col-md-6">
                        {exportFormat === 'docx' && (
                            <>
                                <label htmlFor="fontSize" className="form-label">Font Size:</label>
                                <input 
                                    type="number" 
                                    className="form-control" 
                                    id="fontSize" 
                                    value={fontSize}
                                    onChange={(e) => setFontSize(Number(e.target.value))}
                                    min="8"
                                    max="72"
                                />
                            </>
                        )}
                        {exportFormat === 'txt' && (
                            <>
                                <label htmlFor="paragraphWidth" className="form-label">Paragraph Width:</label>
                                <input 
                                    type="number" 
                                    className="form-control" 
                                    id="paragraphWidth" 
                                    value={paragraphWidth}
                                    onChange={(e) => setParagraphWidth(Number(e.target.value))}
                                    min="40"
                                    max="100"
                                />
                            </>
                        )}
                    </div>
                </div>

                <div className="text-center mb-3">
                    <button
                        type="button"
                        className={`btn ${hasConfidence ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setHasConfidence(!hasConfidence)}
                    >
                        Display confidence: {hasConfidence ? 'true' : 'false'}
                    </button>
                </div>

                <div className="text-center">
                    <button type="submit" className="btn btn-primary">Upload</button>
                </div>
            </form>

            {error && (
                <div className="alert alert-danger mt-3" role="alert">
                    {error}
                </div>
            )}

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
