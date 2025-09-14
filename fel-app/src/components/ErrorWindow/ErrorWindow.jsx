import React from "react";
import './ErrorWindow.css';

const ErrorWindow = ({message, showError}) => {
    
    return <div className={`error-window ${showError ? 'show' : 'hide'}`}>
                {message}
            </div>

};

export default ErrorWindow;
