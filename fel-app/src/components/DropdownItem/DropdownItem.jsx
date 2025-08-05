import React from "react";
import "./DropdownItem.css";

const DropdownItem = ({children, onClick}) => {
    return (
        <div className='dropdownItem' onClick={onClick}>
            {children}
        </div>
        );
}

export default DropdownItem;
