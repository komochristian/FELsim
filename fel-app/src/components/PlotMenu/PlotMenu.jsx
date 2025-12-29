import React, { useRef, useState } from 'react';
import Overlay from 'react-bootstrap/Overlay';
import { Container, Row, Button, Col } from "react-bootstrap";

const PlotMenu = ({ saveFig, setSelectedMenu }) => {
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
                    zIndex: 10,
                    border: '1px solid #000000',
                    backgroundColor: '#FFFFFF',

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
                    className="bg-white border border-dark rounded shadow-sm p-1 ms-1 d-inline-block"
                    style={{ width: "fit-content", maxWidth: "100vw" }}
                    onMouseEnter={() => setShowPlotMenu(true)}
                    onMouseLeave={() => setShowPlotMenu(false)}
                >
                    <Row className='d-inline-block'>
                        <Col xs='auto'>
                            <Button 
                                className='me-2 h-auto'
                                onClick={() => saveFig()}
                            >
                                Download PNG
                            </Button>
                            <Button 
                                className='me-2 h-auto'
                                onClick={() => setSelectedMenu('s_pos_select')}>
                                Go to S position
                            </Button>
                        </Col>
                    </Row>
                </Container>
            </Overlay>
        </>
    )
};

export default PlotMenu;