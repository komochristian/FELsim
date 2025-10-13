import React from 'react';
import {Tooltip} from 'react-tooltip';

const FloatingInfoButton = () => {
  return (
    <div>
      {/* Floating Button */}
      <button
        data-tooltip-id="info-tooltip"
        data-tooltip-html="To get started, load a beamline and simulate with desired settings<br />
                           Tips:<br />
                           1. Click on the graph to view the Z position.<br />
                           2, Use the dropdown to change the Twiss parameter displayed.<br />
                           3. Enable Scroll lets you hover over the graph to scroll through the beamline.<br />
                           4. Click on legend items to toggle visibility.<br />
                            <br />
                           For questions, contact: komochristian@gmail.com"
        data-tooltip-place="left"
        style={{
          position: 'fixed',
          top: '10px',
          right: '10px',
          backgroundColor: '#000000',
          color: '#fff',
          border: 'none',
          borderRadius: '50%',
          width: '40px',
          height: '40px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        }}
      >
        ? 
      </button>

      {/* Tooltip */}
      <Tooltip id="info-tooltip" />
    </div>
  );
};

export default FloatingInfoButton;