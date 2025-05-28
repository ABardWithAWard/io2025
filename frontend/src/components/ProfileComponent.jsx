import React from 'react';
import NavbarComponent from './NavbarComponent';

function ProfileComponent() {
    return (
        <div>
            <NavbarComponent />
            <div className="container mt-4">
                <h1>Profile Page</h1>
                {/* Profile content will be added here */}
            </div>
        </div>
    );
}

export default ProfileComponent; 