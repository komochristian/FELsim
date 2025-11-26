import 'bootstrap/dist/css/bootstrap.min.css';
import {Row, Col, Card, Container, Button, Form} from 'react-bootstrap';
import './ModalContent.css';
import { useState } from 'react';
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";
import 'bootstrap/dist/css/bootstrap.min.css';
import { API_ROUTE, PRIVATEVARS, MODALPRIVATEVARS, TWISS_OPTIONS } from '../../constants';
import ParameterGraph from '../ParameterGraph/ParameterGraph';
import Select from 'react-select';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const ModalContent = ({ beamline, showErrorWindow }) => {
        const schema = yup
    .object()
    .shape({
        "s-pos": yup
            .number()
            .required('S position is required')
            .min(0, 'S position must be non-negative')
            .max(beamline[beamline.length - 1].endPos, 'S needs to be within the beamline'),
        "target_parameter": yup.string().required('Parameter selection is required'),
        "min": yup.number().default(0).min(0, 'min must be non-negative'),
        "max": yup.number().default(10),
        "custom_step": yup.number().default(1).moreThan(0, 'Step size must be positive'),
    })
    .required()

    // State to track hovered rectangle and tooltip visibility
    const [hovered, setHovered] = useState(null);
    const [tooltipStyle, setTooltipStyle] = useState({ display: 'none' });
    const [beamElementSelected, setSelectedElement] = useState(null);
    const [beamIndex, setBeamIndex] = useState(null);
    const [plotData, setPlotData] = useState(null);
    const [simulatedData, setSimulatedData] = useState(null);
    const [currentTwissParam, setCurrentTwiss] = useState({value: 'Envelope\\ E (mm)',
                                                           label: 'Envelope\\ E (mm)',
                                                           modal_val: 'envelope'});

    const {
        register,
        handleSubmit,
        reset,
        formState: { errors },
      } = useForm({
        resolver: yupResolver(schema),
      });

      const onSubmit = async (data) => {
        // console.log('Form submitted:', data);

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

        // console.log('cleanedList:', cleanedList);

        const cleanedData = {
            beam_index: beamIndex,
            target_parameter: data.target_parameter,
            target_s_pos: data['s-pos'],
            beamline_data: cleanedList,
            min: data['min'],
            max: data['max'],
            custom_step: data['custom_step'],
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
                    const { startPos, endPos, color, name } = item;
                    const adjustedStartPos = 500*startPos/totalLength;
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
                    / >
                    );
                })}
                </svg>
            </div>
        </Row>
        <Row className="justify-content-center">
            <Col md={3}>
                <Row className="mb-2">
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
                </Row>
                <Row>
                    <Card>
                        <Card.Body>
                            <Card.Title>Twiss Parameter</Card.Title>
                            <Select className='select-container'
                                options={TWISS_OPTIONS}
                                value={currentTwissParam}
                                onChange={setCurrentTwiss}
                                getOptionLabel={e => <InlineMath math={e.label} />}
                                getSingleValueLabel={e => <InlineMath math={e.modal_val} />}
                                menuPortalTarget={document.body}
                                menuPosition="fixed"
                                styles={{
                                    menuPortal: base => ({ ...base, zIndex: 9999 })
                                }}
                            />
                        </Card.Body>
                    </Card>
                </Row>
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
                            <Row>
                                <Col>
                                    <Form.Group>
                                        <Form.Label>Min</Form.Label>
                                        <input
                                            type="number"
                                            step="any"
                                            {...register('min')}
                                            min={0}
                                            className={`form-control ${errors['min'] ? 'is-invalid' : ''}`}
                                        />
                                        <div className="invalid-feedback">{errors.min?.message}</div>
                                    </Form.Group>
                                </Col>
                                <Col>
                                    <Form.Group>
                                        <Form.Label>Max</Form.Label>
                                        <input
                                            type="number"
                                            step="any"
                                            {...register('max')}
                                            className={`form-control ${errors['max'] ? 'is-invalid' : ''}`}
                                        />
                                        <div className="invalid-feedback">{errors.max?.message}</div>
                                    </Form.Group>
                                </Col>
                                <Col>
                                    <Form.Group>
                                        <Form.Label>Interval</Form.Label>
                                        <input
                                            type="number"
                                            step="any"
                                            default={1}
                                            {...register('custom_step')}
                                            min={0}
                                            className={`form-control ${errors.custom_step ? 'is-invalid' : ''}`}
                                        />
                                        <div className="invalid-feedback">{errors.custom_step?.message}</div>
                                    </Form.Group>
                                </Col>
                            </Row>
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
                    parameter_name={simulatedData?.target_parameter || ''}
                    twiss_target={currentTwissParam.modal_val || ''}
                />
            </Col>
        </Row>
        </Container>
  );
};

export default ModalContent;