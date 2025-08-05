import "./DropdownButton.css";

import React from "react";
import {FaChevronDown, FaChevronUp} from "react-icons/fa";

const DropdownButton = ({children, toggle, open}) => {
    return <div
            className={`dropdownBtn ${open ? "buttonOpen" : null}`}
            onClick={toggle}>
        {children}
        <span className="dropdownIcon">
           {open ? <FaChevronUp/> : <FaChevronDown />}
        </span>
    </div>;
};

export default DropdownButton;
