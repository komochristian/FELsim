import 'bootstrap/dist/css/bootstrap.min.css';
import {Row, Col, Card} from 'react-bootstrap';
import './ModalContent.css';
import { useState } from 'react';

const ModalContent = ({ beamline }) => {
    // State to track hovered rectangle and tooltip visibility
    const [hovered, setHovered] = useState(null);
    const [tooltipStyle, setTooltipStyle] = useState({ display: 'none' });
    const [beamElementSelected, setSelectedElement] = useState(null);

    // Handle hover over a rectangle
    const handleMouseEnter = (info, event) => {
        setHovered(info);
        setTooltipStyle({
            visible: true,
        });
    };

    // Handle mouse leave
    const handleMouseLeave = () => {
        setHovered(null);
        setTooltipStyle({ 
                            display: 'none',
                            visible: false
                        });
    };

    // Handle click on a rectangle
    const handleClick = (info) => {
        const key = info.target.getAttribute('data-key');
        setSelectedElement(beamline[key]);
        console.log(beamline[key]);
    };
  
    //TODO
    //1. MAKE X BUTTON NOT GET IN THE WAY
    //2. MAKE IT SCROLLABLE IF TOO LONG
    const tempTopKey = Object.keys(beamline[beamline.length-1])[0];
    const totalLength = beamline[beamline.length-1][tempTopKey].endPos;
    return (
        <>
        <svg
            viewBox={`0 0 500 50`} // Define the coordinate system for the SVG
            preserveAspectRatio="none" // Ensure the SVG stretches to fill the container
            width="100%" // Make the SVG responsive to the container's width
            height="100%" // Make the SVG responsive to the container's height
        >
        {beamline.map((item, index) => {
            const topKey = Object.keys(item)[0];
            let { startPos, endPos, color } = item[topKey];
            startPos = 500*startPos/totalLength;
            endPos = 500*endPos/totalLength;
            const width = 500*(endPos - startPos)/totalLength; // Calculate the length of the rectangle
            return (
            <rect
                key={index}
                data-key={index}
                x={startPos} // Position the rectangle based on startPos
                y={0}
                width={width} // Set the width based on the length
                height={20} // Fixed height for all rectangles
                fill={color} // Apply the color dynamically
                onMouseEnter={(e) => handleMouseEnter(`${color}`, e)}
                onMouseLeave={handleMouseLeave}
                onClick={(e) => handleClick(e)}
            / >
            );
        })}
        </svg>
        <div
            style={{
            ...tooltipStyle,
            }}
            className='infoTip'
        >
            {hovered}
        </div>
        <Row>
            <Col>
                <Card>
                    <Card.Body>
                        <Card.Title>Beam element selected</Card.Title>
                        <Card.Text>
                            
                        </Card.Text>
                    </Card.Body>
                </Card>

            </Col>
        </Row>
        </>
  );
};

export default ModalContent;