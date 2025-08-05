import "./DropdownContent.css";
import React from "react";

const DropdownContent = ({children, open}) => {
    return <div className={`dropdownContent ${open ? "dropdownOpen" : null}`}>
         {children}
        </div>;
};

export default DropdownContent;
