import React, { useState, useEffect, useRef } from 'react';
import { Container, Form, Button, Alert, Row, Col } from 'react-bootstrap';

const allowedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'];

const UploadComponent = () => {
  const [errors, setErrors] = useState([]);
  const [files, setFiles] = useState([]);
  const [hasShownPrivacyWarning, setHasShownPrivacyWarning] = useState(false);
  const fileInputRef = useRef(null);
  const dialogRef = useRef(null);

  useEffect(() => {
    fetch("/api/files")
      .then(response => response.json())
      .then(setFiles)
      .catch(err => console.error("Error fetching files:", err));
  }, []);

  const validateFile = () => {
    const fileInput = fileInputRef.current;
    const file = fileInput?.files[0];
    if (file) {
      const fileName = file.name.toLowerCase();
      if (!allowedExtensions.some(ext => fileName.endsWith(ext))) {
        setErrors(["Only image files are allowed."]);
        fileInput.value = ''; // Clear input
        return false;
      }
    }
    return true;
  };

  const handleFormSubmit = (e) => {
    if (!hasShownPrivacyWarning) {
      e.preventDefault();
      dialogRef.current?.showModal();
      return;
    }
    if (!validateFile()) {
      e.preventDefault();
    }
  };

  const handleContinue = () => {
    setHasShownPrivacyWarning(true);
    dialogRef.current?.close();
    document.getElementById('uploadForm').submit();
  };

  const handleCancel = () => {
    dialogRef.current?.close();
  };

  return (
    <Row className="align-items-center h-50">
      <Col></Col>
      <Col>
        <div className="mb-3">
          <form method="POST" encType="multipart/form-data" onSubmit={handleFormSubmit} id="uploadForm">
            <Form.Label htmlFor="formFile" column={""}>Upload a File</Form.Label>
            <Form.Control type="file" name="file" accept={allowedExtensions.join(',')} required ref={fileInputRef} id="formFile" />

            {errors.length > 0 && (
              <Alert variant="danger">
                {errors.map((err, idx) => <div key={idx}>{err}</div>)}
              </Alert>
            )}

            <Button type="submit" className="btn btn-secondary mt-3" id="uploadButton">Upload</Button>
          </form>
        </div>

        <Container className="mt-5 p-0">
          <Form.Label column={""}>Uploaded Files</Form.Label>
          <ul className="list-group" id="file-list">
            {files.map((file, idx) => (
              <li key={idx} className="list-group-item">
                <a href={`/media/${file}`} download>{file}</a>
              </li>
            ))}
          </ul>
        </Container>

        <dialog id="privacyDialog" ref={dialogRef}>
          <h3>Privacy Warning</h3>
          <p>Please do not upload any private or sensitive information.</p>
          <div>
            <button onClick={handleContinue}>Continue</button>
            <button onClick={handleCancel}>Cancel</button>
          </div>
        </dialog>
      </Col>
      <Col></Col>
    </Row>
  );
};

export default UploadComponent;
