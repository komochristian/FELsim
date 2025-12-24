import React from 'react';
import {Tooltip} from 'react-tooltip';

const FloatingInfoButton = () => {
  return (
    <div style={{ position: 'absolute', top: '10px', right: '10px', zIndex: 100}}>
      {/* Floating Button */}
      <button
        data-tooltip-id="info-tooltip"
        data-tooltip-html="To get started, load a beamline and simulate with desired settings<br />
                           Tips:<br />
                            - Click on the line graph to view particle data at S position.<br />
                            - Click on legend items to toggle visibility.<br /><br/>

                           Settings Guide:<br />
                            - Use the dropdown to change the Twiss parameter displayed.<br />
                            - Enable Scroll lets you hover over the graph to scroll through the beamline.<br /><br />
                            
                            Optimization Guide:<br />
                             - Visualize and plot twiss parameters as a function of a beam element's parameter.<br />
                             - Select position you want to plot twiss parameter at.<br />
                             - select beamline element and the parameter you want to optimize.
                             - adjust plot settings as needed.<br />
                             
                             <br/>
                           For questions, contact: komochristian@gmail.com"
        data-tooltip-place="left"
        style={{
          position: 'absolute',
          top: '1px',
          right: '3px',
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