import React, { useState } from 'react';
import './EditWindow.css';
import ReactDOM from 'react-dom';

const EditWindow = ({open, parameters, index, onChange, position, PRIVATEVARS}) => {
  if (!open) return null;

  const content = (
    <div
      className="windowContent windowContentOpen"
      style={{
        position: 'fixed',
        top: position.top,
        left: position.left,
      }}
    >
      <div className="paramGrid">
        {Object.entries(parameters).map(([key, value]) => {
          if (key === "type" || key === "id" || PRIVATEVARS.includes(key)) return null;

          return (
            <div key={key} className="paramItem">
              <label className="paramLabel">{key}</label>
              <input
                className="paramInput"
                type="number"
                value={value}
                onChange={(e) => onChange(index, key, parseFloat(e.target.value))}
              />
            </div>
          );
        })}
      </div>
    </div>
  );

  return ReactDOM.createPortal(content, document.body);
};

export default EditWindow;
