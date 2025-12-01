import React, { useState } from 'react';
import { Row } from 'react-bootstrap';
import "./ScrollableBeam.css";

const ScrollableBeam = ({ beamline }) => {
    const [tooltipStyle, setTooltipStyle] = useState({ display: 'none' });
    const [hovered, setHovered] = useState(null);
    const [beamIndex, setBeamIndex] = useState(null);
    const [beamElementSelected, setSelectedElement] = useState(null);

    const totalLength = beamline[beamline.length-1].endPos;

    const handleMouseEnter = (startPos, name,  event) => {
        setHovered(`${name} (Start: ${Math.round(startPos * 10000) / 10000} m) (index: ${event.target.getAttribute('data-key')})`);
        setTooltipStyle({
            visible: true,
        });
    };

    const handleMouseLeave = () => {
        setHovered(null);
        setTooltipStyle({ 
                            display: 'none',
                            visible: false
                        });
    };

    const handleClick = (info) => {
        const key = info.target.getAttribute('data-key');
        setBeamIndex(key);
        setSelectedElement(beamline[key]);
        // console.log(beamline[key]);
    };
    return <>
        <div
            style={{...tooltipStyle,}}
            className='infoTip'
        >
            {hovered}
        </div>
        <Row className="mb-3">
            <div className="modalWrapper">
                <svg
                    viewBox={`0 0 500 50`} // Define the coordinate system for the SVG
                    preserveAspectRatio="none" 
                    //NOTE MAKE THIS ADJUSTABLE FOR THE USER BELOW
                    width={`${beamline.length * 50}px`}
                    height="100%" 
                >
                {beamline.map((item, index) => {
                    const { startPos, endPos, color, name } = item;
                    const adjustedStartPos = 500 * startPos/totalLength;
                    const adjustedEndPos = 500 * endPos/totalLength;
                    const width = 500*(adjustedEndPos - adjustedStartPos)/totalLength;
                    return (
                    <rect
                        key={index}
                        data-key={index}
                        x={adjustedStartPos}
                        y={0}
                        width={width}
                        height={20}
                        fill={color}
                        onMouseEnter={(e) => handleMouseEnter(startPos, name, e)}
                        onMouseLeave={handleMouseLeave}
                        onClick={(e) => handleClick(e)}
                    />
                    );
                })}
                </svg>
            </div>
        </Row>
    </>;
    };