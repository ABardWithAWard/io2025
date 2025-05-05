import React, { useState } from 'react';
import { Navbar, Nav, Container, Button, Modal, Tab, Tabs, Form, Alert } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';
import './NavbarComponent.css'; // Custom CSS for styling

function NavbarComponent() {
  const [showModal, setShowModal] = useState(false);
  const [activeTab, setActiveTab] = useState('login');
  const [messages, setMessages] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const navigate = useNavigate();

  const handleModalClose = () => setShowModal(false);
  const handleModalShow = () => setShowModal(true);
  const handleTabSelect = (k) => setActiveTab(k);

  return (
    <>
      <Navbar bg="light" expand="lg" className="bg-body-tertiary">
        <Container>
          <Navbar.Brand as={Link} to="/">Aplikacja OCR</Navbar.Brand>
          <Navbar.Toggle aria-controls="navbarSupportedContent" />
          <Navbar.Collapse id="navbarSupportedContent">
            <Nav className="me-auto mb-2 mb-lg-0">
              <Nav.Link as={Link} to="/contact">Kontakt</Nav.Link>
              <Nav.Link as={Link} to="/admin">Panel administracji</Nav.Link>
            </Nav>
            <Nav className="ml-auto">
              {isAuthenticated ? (
                <>
                  <span className="navbar-text me-3">{userEmail}</span>
                  <Button variant="outline-primary" as={Link} to="/logout">Logout</Button>
                </>
              ) : (
                <Button variant="outline-primary" onClick={handleModalShow}>Login</Button>
              )}
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {/* Messages */}
      {messages.length > 0 && (
        <div className="messages">
          {messages.map((msg, idx) => (
            <Alert key={idx} variant="danger" dismissible onClose={() => setMessages([])}>
              {msg}
            </Alert>
          ))}
        </div>
      )}

      {/* Modal */}
      <Modal show={showModal} onHide={handleModalClose} centered>
        <Modal.Header closeButton>
          <Modal.Title>{activeTab === 'login' ? 'Login' : 'Register'}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Tabs activeKey={activeTab} onSelect={handleTabSelect} className="mb-3">
            <Tab eventKey="login" title="Login">
              <Form action="/login" method="POST">
                <Form.Group className="mb-3" controlId="loginEmail">
                  <Form.Label>Email</Form.Label>
                  <Form.Control type="email" name="email" required />
                </Form.Group>
                <Form.Group className="mb-3" controlId="loginPassword">
                  <Form.Label>Password</Form.Label>
                  <Form.Control type="password" name="password" required />
                </Form.Group>
                <Button variant="primary" type="submit" className="w-100 mb-2">Login</Button>
                <Button variant="light" className="w-100 d-flex align-items-center justify-content-center border">
                  <img src="https://www.google.com/favicon.ico" alt="Google" width="18" className="me-2" />
                  Login with Google
                </Button>
              </Form>
            </Tab>
            <Tab eventKey="register" title="Register">
              <Form action="/register" method="POST">
                <Form.Group className="mb-3" controlId="registerEmail">
                  <Form.Label>Email</Form.Label>
                  <Form.Control type="email" name="email" required />
                </Form.Group>
                <Form.Group className="mb-3" controlId="registerPassword">
                  <Form.Label>Password</Form.Label>
                  <Form.Control type="password" name="password" required />
                </Form.Group>
                <Form.Group className="mb-3" controlId="registerConfirmPassword">
                  <Form.Label>Confirm Password</Form.Label>
                  <Form.Control type="password" name="confirm_password" required />
                </Form.Group>
                <Button variant="primary" type="submit" className="w-100 mb-2">Register</Button>
                <Button variant="light" className="w-100 d-flex align-items-center justify-content-center border">
                  <img src="https://www.google.com/favicon.ico" alt="Google" width="18" className="me-2" />
                  Register with Google
                </Button>
              </Form>
            </Tab>
          </Tabs>
        </Modal.Body>
      </Modal>
    </>
  );
}

export default NavbarComponent;
