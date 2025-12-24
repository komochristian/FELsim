import React, { useRef, useState } from 'react';
import Overlay from 'react-bootstrap/Overlay';
import { Container, Row, Button, Col } from "react-bootstrap";

const PlotMenu = ({ saveFig }) => {
    const targetRef = useRef(null);
    const [showPlotMenu, setShowPlotMenu] = useState(false);

    return (
        <>
            <button
                ref={targetRef}
                onMouseEnter={() => setShowPlotMenu(true)}
                onMouseLeave={() => setShowPlotMenu(false)}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    margin: '8px',
                    zIndex: 10
                }}
            >
                <i className="fas fa-bars"></i>
            </button>
            <Overlay
                show={showPlotMenu}
                target={targetRef.current}
                placement="right"
            >
                <Container 
                    className="bg-white border border-dark rounded shadow-sm p-1 ms-3"
                    onMouseEnter={() => setShowPlotMenu(true)}
                    onMouseLeave={() => setShowPlotMenu(false)}
                    style={{
                        width: '400px'
                    }}
                >
                    <Row>
                        <Col>
                            <Button 
                                className='me-2 h-auto'
                                onClick={() => saveFig()}
                            >
                                Download PNG
                            </Button>
                            <Button className='me-2 h-auto'>
                                test
                            </Button>
                        </Col>
                    </Row>
                </Container>
            </Overlay>
        </>
    )
};

export default PlotMenu;