import React, { useState } from 'react';
import './BeamSegment.css';

const BeamSegment = ({ name, index, onRename, onDelete }) => {
    const [editing, setEditing] = useState(false);
    const [value, setValue] = useState(name);

    const handleSave = () => {
        onRename(index, value);
        setEditing(false);
    };

    return (
        <div className="beamSegment">
            {editing ? (
                <>
                    <input 
                        value={value} 
                        onChange={(e) => setValue(e.target.value)} 
                    />
                    <button onClick={handleSave}>Save</button>
                </>
            ) : (
                <>
                    <span>{name}</span>
                    <button onClick={() => setEditing(true)}>Edit</button>
                </>
            )}
            <button onClick={() => onDelete(index)}>Delete</button>
        </div>
    );
};

export default BeamSegment;

