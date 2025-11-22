import 'bootstrap/dist/css/bootstrap.min.css';
import {Row, Col, Card, Container, Button, Form} from 'react-bootstrap';
import './ModalContent.css';
import { useState } from 'react';
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";
import 'bootstrap/dist/css/bootstrap.min.css';
import { API_ROUTE, PRIVATEVARS, MODALPRIVATEVARS } from '../../constants';
import ParameterGraph from '../ParameterGraph/ParameterGraph';

const ModalContent = ({ beamline, twissOptions, showErrorWindow }) => {
        const schema = yup
    .object()
    .shape({
        "s-pos": yup
            .number()
            .required('S position is required')
            .min(0, 'S position must be non-negative')
            .max(beamline[beamline.length - 1].endPos, 'S needs to be within the beamline'),
        "target_parameter": yup.string().required('Parameter selection is required'),
        "twiss_target": yup.string().required('Select a twiss parameter to plot'),
    })
    .required()

    // State to track hovered rectangle and tooltip visibility
    const [hovered, setHovered] = useState(null);
    const [tooltipStyle, setTooltipStyle] = useState({ display: 'none' });
    const [beamElementSelected, setSelectedElement] = useState(null);
    const [beamIndex, setBeamIndex] = useState(null);
    const [plotData, setPlotData] = useState(null);
    const [simulatedData, setSimulatedData] = useState(null);

    const {
        register,
        handleSubmit,
        reset,
        formState: { errors },
      } = useForm({
        resolver: yupResolver(schema),
      });

      const onSubmit = async (data) => {
        console.log('Form submitted:', data);

        const cleanedList = beamline.map(obj => {
            const key = obj.name;
            const cleanedParams = Object.fromEntries(
              Object.entries(obj).filter(([p]) => !PRIVATEVARS.includes(p))
            );
            return {
                segmentName: key,
                parameters: cleanedParams
            };
        });

        console.log('cleanedList:', cleanedList);

        const cleanedData = {
            beam_index: beamIndex,
            target_parameter: data.target_parameter,
            target_s_pos: data['s-pos'],
            beamline_data: cleanedList,
            twiss_target: data.twiss_target
        }
        setSimulatedData(cleanedData);

        const res = await fetch(API_ROUTE + '/plot-parameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(cleanedData, null, 2),
        });
        // console.log('Res:', res);
        if (!res.ok) {
            const errorData = await res.json();
            showErrorWindow(errorData.detail || errorData);
            return 
        }
        const responseData = await res.json();
        console.log('Response Data:', responseData);
        setPlotData(responseData);
      };

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
        setBeamIndex(key);
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
            <Col md={3}>
                <Card>
                    <Card.Body>
                        <Form onSubmit={handleSubmit(onSubmit)}>
                            <Form.Group>
                                <Form.Label>Enter a position along the beamline to optimize</Form.Label>
                                <input
                                    type="number"
                                    step="any"
                                    {...register('s-pos')}
                                    min={beamElementSelected?.endPos || 0}
                                    max={beamline[beamline.length - 1].endPos}
                                    className={`form-control ${errors['s-pos'] ? 'is-invalid' : ''}`}
                                />
                                <div className="invalid-feedback">{errors['s-pos']?.message}</div>
                            </Form.Group>
                            <Form.Group>
                                {beamElementSelected ? (
                                    <>
                                      <Form.Label>Beam Parameter to Plot:</Form.Label>
                                      <select {...register('target_parameter')} className={`form-control ${errors.target_parameter ? 'is-invalid' : ''}`}>
                                        {Object.entries(beamElementSelected).map(([key, value]) => {
                                                if (!MODALPRIVATEVARS.includes(key)) {
                                                    return <option value={key}>{key}</option>;
                                                }
                                            }
                                        )}
                                      </select>
                                      <div className="invalid-feedback">{errors.target_parameter?.message}</div>
                                    </>
                                ) : (
                                    <>
                                        <Form.Label className='text-danger'>Please select a beam element</Form.Label> 
                                        <input type="hidden" {...register('target_parameter')} value='' />
                                        <div className="invalid-feedback">{errors.target_parameter?.message}</div>
                                    </>
                                )}
                            </Form.Group>
                            <Form.Group>
                                <Form.Label>Select Twiss Parameter to Plot:</Form.Label>
                                <select {...register('twiss_target')} className={`form-control ${errors.twiss_target ? 'is-invalid' : ''}`}>
                                    {twissOptions.map(opt => (
                                        <option key={opt.modal_val} value={opt.modal_val}>
                                            {opt.modal_val}
                                        </option>
                                    ))}
                                </select>
                                <div className="invalid-feedback">{errors.twiss_target?.message}</div>
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
            <Col md={6}>
                <ParameterGraph
                    data={plotData}
                    parameter_name={simulatedData?.target_parameter | ''}
                    twiss_target={simulatedData?.twiss_target || ''}
                />
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