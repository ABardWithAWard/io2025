import React, {useState, useEffect, useRef} from 'react';
import { Navbar, Nav, Container, Button, Modal, Tab, Tabs, Form, Alert } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import './NavbarComponent.css';

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_OAUTH2_CLIENT_ID;

// Main navigation component that handles navigation and user login
function NavbarComponent() {
  const [showModal, setShowModal] = useState(false);
  const [showAdminAccessModal, setShowAdminAccessModal] = useState(false);
  const [activeTab, setActiveTab] = useState('login');
  const [messages, setMessages] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [setIsStaff] = useState(false);
  const navigate = useNavigate();
  const { getCsrfToken, checkAuthentication, refreshCsrfToken, logout, userEmail } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: ''
  });

  useEffect(() => {
    // Check authentication status on component mount
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const isAuth = await checkAuthentication();
      if (isAuth) {
        setIsAuthenticated(true);
        await refreshCsrfToken();
      } else {
        setIsAuthenticated(false);
        setIsStaff(false);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
    }
  };

  const handleAdminClick = async (e) => {
    e.preventDefault();
    try {
      const isAuth = await checkAuthentication();
      if (isAuth) {
        navigate('/admin');
      } else {
        setShowAdminAccessModal(true);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      setShowAdminAccessModal(true);
    }
  };

  // Basic modal logic which clears form and hides/shows modals themselves
  const handleModalClose = () => {
    setShowModal(false);
    setMessages([]);
  };

  const handleModalShow = () => setShowModal(true);

  const handleTabSelect = (k) => {
    setActiveTab(k);
    setMessages([]);
  };

  // Generic input handler for updating form state in React
  // e.target -> HTML input element which just triggered this event
  // name and value are attributes of it
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    })); // Updates form data without losing other fields, ...prev copies previous state
    // and then updates the field that matches name
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    let formData;
    try {
      const response = await fetch('/api/login/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        credentials: 'include',
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
        })
      });

      const data = await response.json();

      if (response.ok) {
        setIsAuthenticated(true);
        handleModalClose();
        navigate('/'); // After login navigate to base page to avoid silly errors
      } else {
        setMessages([data.error || 'Login failed']);
      }
    } catch (error) {
      console.error('Login error:', error);
      setMessages(['An error occurred during login']);
    }
  };

  // Same as login, literally same logic but one additional field
  const handleRegister = async (e) => {
    e.preventDefault();
    let formData;

    if (formData.password !== formData.confirmPassword) {
      setMessages(['Passwords do not match']);
      return;
    }

    try {
      const response = await fetch('/api/register/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        credentials: 'include',
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          confirmPassword: formData.confirmPassword
        })
      });

      const data = await response.json();

      if (response.ok) {
        setIsAuthenticated(true);
        handleModalClose();
        navigate('/');
      } else {
        setMessages([data.error || 'Registration failed']);
      }
    } catch (error) {
      console.error('Registration error:', error);
      setMessages(['An error occurred during registration']);
    }
  };

  // Same logic as login, but different implementation on backend
  window.handleGoogleLogin = async (response) => {
    const idToken = response.credential;
    try {
      const res = await fetch('/api/google-auth/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        credentials: 'include',
        body: JSON.stringify({ idToken }),
      });

      const data = await res.json();

      if (res.ok) {
        setIsAuthenticated(true);
        handleModalClose();
        window.location.href = '/'; // This will cause a full page refresh, fixes bug with e-mail not appearing
      } else {
        setMessages([data.error || 'Google login failed']);
      }
    } catch (err) {
      console.error('Google login error:', err);
      setMessages(['An error occurred during Google login']);
    }
  };

  // Same logic as login
  const handleLogout = async () => {
    try {
      const response = await fetch('/api/logout/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        credentials: 'include'
      });

      if (response.ok) {
        setIsAuthenticated(false);
        navigate('/');
      } else {
        setMessages(['Logout failed']);
      }
    } catch (error) {
      console.error('Logout error:', error);
      setMessages(['An error occurred during logout']);
    }
  };

  return (
    <>
      <Navbar bg="light" expand="lg" className="bg-body-tertiary">
        <Container>
          <Navbar.Brand as={Link} to="/">Aplikacja OCR</Navbar.Brand>
          <Navbar.Toggle aria-controls="navbarSupportedContent" />
          <Navbar.Collapse id="navbarSupportedContent">
            <Nav className="me-auto mb-2 mb-lg-0">
              <Nav.Link as={Link} to="/contact">Kontakt</Nav.Link>
              <Nav.Link onClick={handleAdminClick}>Admin</Nav.Link>
            </Nav>
            <Nav className="ml-auto">
              {isAuthenticated ? (
                <>
                  <Nav.Link as={Link} to="/profile" className="me-3">{userEmail}</Nav.Link>
                  <Button variant="outline-primary" onClick={handleLogout}>Logout</Button>
                </>
              ) : (
                <Button variant="outline-primary" onClick={handleModalShow}>Login</Button>
              )}
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {/* Admin Access Modal */}
      <Modal show={showAdminAccessModal} onHide={() => setShowAdminAccessModal(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Admin Access Required</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Log in with admin account to access admin panel</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowAdminAccessModal(false)}>
            Close
          </Button>
          {!isAuthenticated && (
            <Button variant="primary" onClick={() => {
              setShowAdminAccessModal(false);
              setShowModal(true);
            }}>
              Login
            </Button>
          )}
        </Modal.Footer>
      </Modal>

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
              <Form onSubmit={handleLogin}>
                <Form.Group className="mb-3" controlId="loginEmail">
                  <Form.Label>Email</Form.Label>
                  <Form.Control 
                    type="email" 
                    name="email" 
                    value={formData.email}
                    onChange={handleInputChange}
                    required 
                  />
                </Form.Group>
                <Form.Group className="mb-3" controlId="loginPassword">
                  <Form.Label>Password</Form.Label>
                  <Form.Control 
                    type="password" 
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    required 
                  />
                </Form.Group>
                <Button variant="primary" type="submit" className="w-100 mb-2">Login</Button>
                <GoogleSignInButton clientId={GOOGLE_CLIENT_ID} />

              </Form>
            </Tab>
            <Tab eventKey="register" title="Register">
              <Form onSubmit={handleRegister}>
                <Form.Group className="mb-3" controlId="registerEmail">
                  <Form.Label>Email</Form.Label>
                  <Form.Control 
                    type="email" 
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required 
                  />
                </Form.Group>
                <Form.Group className="mb-3" controlId="registerPassword">
                  <Form.Label>Password</Form.Label>
                  <Form.Control 
                    type="password" 
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    required 
                  />
                </Form.Group>
                <Form.Group className="mb-3" controlId="registerConfirmPassword">
                  <Form.Label>Confirm Password</Form.Label>
                  <Form.Control 
                    type="password" 
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    required 
                  />
                </Form.Group>
                <Button variant="primary" type="submit" className="w-100 mb-2">Register</Button>
                <GoogleSignInButton clientId={GOOGLE_CLIENT_ID} />
              </Form>
            </Tab>
          </Tabs>
          <div id="google-button" style={{ display: 'none' }}></div>
        </Modal.Body>
      </Modal>
    </>
  );
}

export default NavbarComponent;

const GoogleSignInButton = ({ clientId }) => {
  const buttonDiv = useRef();

  useEffect(() => {
    if (!window.handleGoogleLogin) {
      console.log("Callback is okay");
      window.handleGoogleLogin = async (response) => {
        const idToken = response.credential;
        if (!idToken) return;

        const res = await fetch('/api/google-auth/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({ idToken }),
        });

        const data = await res.json();
        if (res.ok) {
          // success handling here
        } else {
          // show error message
        }
      };
    }

    if (window.google && buttonDiv.current) {
      console.log("initialized");
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: window.handleGoogleLogin,
        ux_mode: 'popup',
      });

      window.google.accounts.id.renderButton(buttonDiv.current, {
        theme: 'outline',
        size: 'large',
        text: 'signin_with',
      });
    }
  }, [clientId]);

  return <div ref={buttonDiv}></div>;
};