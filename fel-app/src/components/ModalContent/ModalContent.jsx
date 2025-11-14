import 'bootstrap/dist/css/bootstrap.min.css';
import {Row, Col, Card, Container, Button, Form} from 'react-bootstrap';
import './ModalContent.css';
import { useState } from 'react';
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";


const schema = yup
  .object()
  .shape({
    "z-pos": yup.number().required('Z position is required'),
  })
  .required()

const ModalContent = ({ beamline }) => {
    const {
        register,
        handleSubmit,
        reset,
        formState: { errors },
      } = useForm({
        resolver: yupResolver(schema),
      });

      const onSubmit = (data) => {
        console.log('Form submitted:', data);
      };

    // State to track hovered rectangle and tooltip visibility
    const [hovered, setHovered] = useState(null);
    const [tooltipStyle, setTooltipStyle] = useState({ display: 'none' });
    const [beamElementSelected, setSelectedElement] = useState(null);

    // Handle hover over a rectangle
    const handleMouseEnter = (startPos, name,  event) => {
        console.log(event)
        setHovered(`${name} (Start: ${Math.round(startPos * 10000) / 10000} m) (index: ${event.target.getAttribute('data-key')})`);
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
    const totalLength = beamline[beamline.length-1].endPos;
    return (
        <Container>
        <div
            style={{
            ...tooltipStyle,
            }}
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
                    let { startPos, endPos, color, name } = item;
                    startPos = 500*startPos/totalLength;
                    endPos = 500*endPos/totalLength;
                    const width = 500*(endPos - startPos)/totalLength;
                    return (
                    <rect
                        key={index}
                        data-key={index}
                        x={startPos}
                        y={0}
                        width={width}
                        height={20}
                        fill={color}
                        onMouseEnter={(e) => handleMouseEnter(startPos, name, e)}
                        onMouseLeave={handleMouseLeave}
                        onClick={(e) => handleClick(e)}
                    / >
                    );
                })}
                </svg>
            </div>
        </Row>
        <Row className="justify-content-center">
            <Col md={3}>
                <Card>
                    <Card.Body>
                        <Card.Title>Beam element selected</Card.Title>
                        <Card.Text> 
                            {beamElementSelected ? (
                                <div>
                                    <p><strong>{beamElementSelected.name}</strong></p>
                                    <p><strong>Start Position:</strong> {Math.round(beamElementSelected.startPos * 10000) /10000} m</p>
                                    <p><strong>End Position:</strong> {Math.round(beamElementSelected.endPos * 10000) /10000} m</p>
                                </div>
                            ) : (
                                <p>No beam element selected.</p>
                            )}

                        </Card.Text>
                    </Card.Body>
                </Card>
            </Col>
            <Col>
                <Card md={3}>
                    <Card.Body>
                        <Form onSubmit={handleSubmit(onSubmit)}>
                            <Form.Group>
                                <Form.Label>Enter a position along the beamline to optimize</Form.Label>
                                <input
                                    type="number"
                                    {...register('z-pos')}
                                    className={`form-control ${errors.note ? 'is-invalid' : ''}`}
                                />
                                {/* ADD ERROR THORWING NEXT */}
                            </Form.Group>
                            <Form.Group className="form-group">
                                <Row className="pt-3">
                                    <Col>
                                        <Button type="submit" variant="primary">
                                            Submit
                                        </Button>
                                    </Col>
                                </Row>
                            </Form.Group>
                        </Form>
                    </Card.Body>
                </Card>
            </Col>
        </Row>
        </Container>
  );
};

{/* <Card.Body>
              <Form onSubmit={handleSubmit(onSubmit)}>
                <Form.Group>
                  <Form.Label>Note</Form.Label>
                  <input
                    type="text"
                    {...register('note')}
                    className={`form-control ${errors.note ? 'is-invalid' : ''}`}
                  />
                  <div className="invalid-feedback">{errors.note?.message}</div>
                </Form.Group>
                <input type="hidden" {...register('owner')} value={currentUser} />
                <input type="hidden" {...register('contactId')} value={contactId} />
                <Form.Group className="form-group">
                  <Row className="pt-3">
                    <Col>
                      <Button type="submit" variant="primary">
                        Submit
                      </Button>
                    </Col>
                    <Col>
                      <Button type="button" onClick={() => reset()} variant="warning" className="float-right">
                        Reset
                      </Button>
                    </Col>
                  </Row>
                </Form.Group>
              </Form>
            </Card.Body> */}

export default ModalContent;