import React, { useState } from 'react';
import { useAuth } from '../AuthContext';

const ContactPage = () => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        message: ''
    });
    const { getCsrfToken } = useAuth();

    const handleSubmit = async (event) => {
        event.preventDefault();
        console.log('Form submitted with data:', formData);

        // Validate form data
        if (!formData.name || !formData.email || !formData.message) {
            console.error('Validation failed: All fields are required');
            alert('Please fill in all fields');
            return;
        }

        const csrfToken = getCsrfToken();
        if (!csrfToken) {
            console.error('No CSRF token available');
            alert('Security token not found. Please refresh the page and try again.');
            return;
        }

        try {
            console.log('Sending POST request to /api/contact/');
            const response = await fetch('/api/contact/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                credentials: 'include',
                body: JSON.stringify({
                    name: formData.name,
                    email: formData.email,
                    message: formData.message
                })
            });

            console.log('Response received:', response.status);
            const data = await response.json();
            console.log('Response data:', data);

            if (response.ok) {
                console.log('Message sent successfully');
                alert('Message sent successfully!');
                setFormData({ name: '', email: '', message: '' });
            } else {
                console.error('Failed to send message:', data.error);
                alert(data.error || 'Failed to send message');
            }
        } catch (error) {
            console.error('Error in handleSubmit:', error);
            alert('Failed to send message');
        }
    };

    const handleChange = (event) => {
        const { name, value } = event.target;
        console.log(`Updating ${name} field:`, value);
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    return (
        <div className="row align-items-center h-50">
            <div className="col"></div>
            <div className="col">
                <form onSubmit={handleSubmit} id="contactForm">
                    <div className="row">
                        <div className="col">
                            <label htmlFor="id-name" className="form-label">Wypełnij formularz zgłoszeniowy</label>
                            <input 
                                type="text" 
                                className="form-control" 
                                name="name" 
                                placeholder="Name" 
                                id="id-name"
                                value={formData.name}
                                onChange={handleChange}
                                required
                            />
                        </div>
                    </div>
                    <div className="row mt-2">
                        <div className="col">
                            <input 
                                type="email"
                                className="form-control"
                                name="email" 
                                placeholder="E-mail"
                                value={formData.email}
                                onChange={handleChange}
                                required
                            />
                        </div>
                    </div>
                    <div className="row mt-2">
                        <div className="col">
                            <textarea 
                                className="form-control" 
                                id="textarea" 
                                rows="3"
                                name="message"
                                placeholder="Your message"
                                value={formData.message}
                                onChange={handleChange}
                                required
                            ></textarea>
                        </div>
                    </div>
                    <button type="submit" className="btn btn-outline-secondary mt-2" id="uploadButton">
                        Submit
                    </button>
                </form>
            </div>
            <div className="col"></div>
        </div>
    );
};

export default ContactPage; 