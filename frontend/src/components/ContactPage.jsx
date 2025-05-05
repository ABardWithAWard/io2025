import React, { useState } from 'react';

const ContactPage = () => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        message: ''
    });

    const handleSubmit = async (event) => {
        event.preventDefault();
        
        try {
            const response = await fetch('/application/contact/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            if (response.ok) {
                alert('Message sent successfully!');
                setFormData({ name: '', email: '', message: '' });
            } else {
                const data = await response.json();
                alert(data.error || 'Failed to send message');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            alert('Failed to send message');
        }
    };

    const handleChange = (event) => {
        const { name, value } = event.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    return (
        <div className="row align-items-center h-50">
            <div className="col"></div>
            <div className="col">
                <form onSubmit={handleSubmit}>
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
                            />
                        </div>
                        <div className="col"></div>
                    </div>
                    <div className="row mt-2">
                        <div className="col">
                            <input 
                                type="text" 
                                className="form-control" 
                                name="email" 
                                placeholder="E-mail"
                                value={formData.email}
                                onChange={handleChange}
                            />
                        </div>
                        <div className="col"></div>
                    </div>
                    <div className="row mt-2">
                        <div className="form-group">
                            <textarea 
                                className="form-control" 
                                id="textarea" 
                                rows="3"
                                name="message"
                                value={formData.message}
                                onChange={handleChange}
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