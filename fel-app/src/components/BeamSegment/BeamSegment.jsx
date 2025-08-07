import React, { useState, useRef } from 'react';
import './BeamSegment.css';
import EditWindow from "../EditWindow/EditWindow";

const BeamSegment = ({ name, params, index, onDelete, onChanges }) => {
  const [editing, setEditing] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const segmentRef = useRef(null);

  const handleToggleEdit = () => {
    if (!editing && segmentRef.current) {
      const rect = segmentRef.current.getBoundingClientRect();
      setPosition({ top: rect.top + 10, left: rect.right + 15 }); // right of the segment
    }
    setEditing(prev => !prev);
  };

  return (
    <div className="beamSegmentRow" ref={segmentRef}>
      <div className="beamSegment">
        <h4>{name}</h4>
        <button onClick={handleToggleEdit}>
          {editing ? 'Close' : 'Edit'}
        </button>
        <button onClick={() => onDelete(index)}>Delete</button>
      </div>

      {editing && (
        <EditWindow
          open={editing}
          parameters={params[name]}
          index={index}
          onChange={onChanges}
          position={position} // 👈 pass the position
        />
      )}
    </div>
  );
};

export default BeamSegment;

