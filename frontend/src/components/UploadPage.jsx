// Main page, used for upload form and various modals which can occur during it
import React, {useEffect, useState} from 'react';
import {useAuth} from '../AuthContext';
import { useNavigate } from 'react-router-dom';

const UploadPage = () => {
    const [showPrivacyDialog, setShowPrivacyDialog] = useState(false);
    const [error, setError] = useState('');
    const [showValidationModal, setShowValidationModal] = useState(false);
    const [validationMessage, setValidationMessage] = useState('');
    const {getCsrfToken, checkAuthentication, userUid} = useAuth();
    const [fontSize, setFontSize] = useState(12);
    const [language, setLanguage] = useState('english');
    const [exportFormat, setExportFormat] = useState('docx');
    const [hasConfidence, setHasConfidence] = useState(false);
    const [paragraphWidth, setParagraphWidth] = useState(80);
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkAuth = async () => {
            await checkAuthentication();
            setLoading(false);
        };

        checkAuth();
    }, [checkAuthentication]);

    const validateFile = async (file) => {
        // Send file to validation endpoint to check image quality
        // Returns validation status and type (dark/bright) if validation fails
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/validate/upload/', {
                method: 'POST',
                body: formData,
                credentials: 'include',
                headers: {
                    'X-CSRFToken': getCsrfToken()
                }
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error validating file:', error);
            return { status: 'error', type: 'validation_error' };
        }
    };

    const handleUploadClick = async (event) => {
        // Instead of uploading right away we ensure that form is well filled
        // and then show a modal which actually handles upload logic
        event.preventDefault();
        const fileInput = document.querySelector('input[type="file"]');
        if (!fileInput || fileInput.files.length === 0) {
            setError('Please select a file to upload');
            return;
        }

        const file = fileInput.files[0];
        const validationResult = await validateFile(file);

        if (validationResult.status === 'success') {
            setShowPrivacyDialog(true);
        } else if (validationResult.status === 'invalid') {
            setValidationMessage(`The image appears to be too ${validationResult.type}. Would you like to proceed anyway?`);
            setShowValidationModal(true);
        } else {
            setError('Error validating file');
        }
    };

    const handleValidationContinue = () => {
        setShowValidationModal(false);
        setShowPrivacyDialog(true);
    };

    const handleValidationCancel = () => {
        setShowValidationModal(false);
    };

    const handlePrivacyContinue = () => {
        // Handle upload logic if user agrees to privacy modal
        setShowPrivacyDialog(false);
        handleFileUpload();
    };

    const handlePrivacyCancel = () => {
        setShowPrivacyDialog(false);
    };

    const handleFileUpload = async () => {
        // Reading form and then appending it for backend api call
        const isAuth = await checkAuthentication();
        const formData = new FormData();
        const fileInput = document.querySelector('input[type="file"]');

        if (fileInput.files.length === 0) {
            setError('Please select a file to upload');
            return;
        }

        try {
            formData.append('file', fileInput.files[0]);
            formData.append('userUid', isAuth ? userUid : null);
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

    if (loading) {
        // Spinner could be added here as well
        return (
            <div className="container mt-4">
                <p>Loading...</p>
            </div>
        );
    }

    return (
        <div className="container mt-4">
            <h2>Upload File</h2>
            {error && (
                <div className="alert alert-danger mb-3" role="alert">
                    {error}
                </div>
            )}
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

                <div className="text-left mb-3">
                    <button
                        type="button"
                        className={`btn ${hasConfidence ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setHasConfidence(!hasConfidence)}
                    >
                        Display confidence: {hasConfidence ? 'true' : 'false'}
                    </button>
                </div>

                <div className="text-left">
                    <button type="submit" className="btn btn-primary">Upload</button>
                </div>
            </form>

            {showPrivacyDialog && (
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
                        <h3>Privacy Warning</h3>
                        <p>Please do not upload any private or sensitive information.</p>
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
                            <button
                                onClick={handlePrivacyContinue}
                                className="btn btn-success"
                            >
                                Continue
                            </button>
                            <button
                                onClick={handlePrivacyCancel}
                                className="btn btn-secondary"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {showValidationModal && (
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
                        <h3>Image Quality Warning</h3>
                        <p>{validationMessage}</p>
                        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
                            <button
                                onClick={handleValidationContinue}
                                className="btn btn-success"
                            >
                                Continue Anyway
                            </button>
                            <button
                                onClick={handleValidationCancel}
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
