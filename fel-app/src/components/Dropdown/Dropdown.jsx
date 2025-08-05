import "./Dropdown.css";
import React from "react";
import DropdownButton from "../DropdownButton/DropdownButton";
import DropdownContent from "../DropdownContent/DropdownContent";
import {useState, useEffect, useRef} from "react";
const Dropdown = ({buttonText, contentText}) => {

    const [open, setOpen] = useState(false);
    const dropdownRef = useRef();
    const toggleDropdown = () => {
        setOpen((open) => !open);
    };

    useEffect(() => {
        const handler = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(
                event.target)) {
                    setOpen(false);
                }
            };
            document.addEventListener("click", handler);
            return () => {
                document.removeEventListener("click", handler);
            };
    }, []);

    return (
        <div className="dropdown" ref={dropdownRef}>
        <DropdownButton toggle={toggleDropdown} open={open}>
            {buttonText}
        </DropdownButton>
        <DropdownContent open={open}>
            {contentText}
        </DropdownContent>
    </div>
    );
};

export default Dropdown;
