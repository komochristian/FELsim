import { Container, Row, Button, Col, Form } from "react-bootstrap";
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import { useState } from "react";
import { useForm } from "react-hook-form";

const ascii_to_greek = (char) => {
    const greekMap = {
        'alpha': 'α',
        'beta': 'β',
        'phi': 'φ',
        'epsilon': 'ε'
    };
    return greekMap[char] || char;
};

const ParticleSettings = ({ setSelectedMenu, submitHelper, twissValues, beamtypeToPass,
    numOfParticles , mev, base_dist, beamSetup}) => {
    const [tabValue, setTabValue] =  useState(beamSetup);
    const {
        register,
        handleSubmit,
        watch,
        reset,
        formState: { isDirty }
      } = useForm({
        defaultValues: {
          beamType: beamtypeToPass,
          customIon: "",
          numParticles: numOfParticles,
          kineticEnergy: mev,
          box_distribution: "gaussian",
          twiss: twissValues,
          base_dist: base_dist
        }
      });
    
    //  TO DO: Add a schema validation with yup 
    const onSubmit = (data) => {
        setSelectedMenu(null);
        data.beam_setup = tabValue;
        submitHelper(data);
    };
    
    const handleChange = (event, newValue) => {
        setTabValue(newValue);
      };

    const getBeamDistributionFromTwiss = async (fileJSON) => {  
    const res =  await fetch(API_ROUTE + '/twiss-to-particles', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(fileJSON, null, 2),
    });
    };

    return (
        <Form onSubmit={handleSubmit(onSubmit)}>
            <Container className="d-flex flex-column align-items-center justify-content-center">

                <h4>Particle Settings</h4>

                {/* Beam Type */}
                <Form.Label className="forLabels">Select Beam type:</Form.Label>
                <Form.Select {...register("beamType")}>
                    <option value="electron">Electron</option>
                    <option value="proton">Proton</option>
                    <option value="otherIon">Other Ion</option>
                    {beamtypeToPass !== "electron" && beamtypeToPass !== "proton" && (
                        <option value={beamtypeToPass}>{beamtypeToPass}</option>
                    )}
                </Form.Select>

                {/* Custom Ion Name */}
                {watch("beamType") === "otherIon" && (
                <Form.Control
                    className="mt-2"
                    placeholder="Ion name"
                    {...register("customIon")}
                />
                )}

                {/* Number of particles */}
                <Form.Label className="forLabels mt-3">
                    Number of particles:
                </Form.Label>
                <Form.Control
                    type="number"
                    min={3}
                    {...register("numParticles", { valueAsNumber: true })}
                />

                {/* Kinetic Energy */}
                <Form.Label className="forLabels mt-3">
                    Kinetic Energy (MeV):
                </Form.Label>
                <Form.Control
                    type="number"
                    min={0}
                    {...register("kineticEnergy", { valueAsNumber: true })}
                />

                {/* Combo box distribution */}
                <Form.Label className="forLabels mt-3">
                    Distribution:
                </Form.Label>
                <Form.Select {...register("box_distribution")}>
                    <option value="gaussian">Gaussian</option>
                    <option value="uniform">Uniform</option>
                </Form.Select>

                {/* Tabs */}
                <Box sx={{ mt: 3 }}>
                <Tabs value={tabValue} onChange={handleChange} centered>
                    <Tab value="twiss" label="Twiss Parameters" />
                    <Tab value="base_dist" label="Base Distributions" />
                    <Tab value="import" label="Import" />
                </Tabs>
                </Box>

                {tabValue === "twiss" && (
                <Container className="mt-3">
                    {Object.entries(twissValues).map(([axis, params]) => (
                        <div key={axis}>
                            <Row>
                            {Object.entries(params).map(([param, value]) => (
                                <Col md={3} key={param}>
                                <Form.Group controlId={`${axis}-${param}`}>
                                    <Form.Label>
                                    {axis}: {ascii_to_greek(param)}
                                    </Form.Label>
                                    <Form.Control
                                    type="number"
                                    {...register(`twiss.${axis}.${param}`)}
                                    />
                                </Form.Group>
                                </Col>
                            ))}
                            </Row>
                        </div>
                    ))}
                </Container>
                )}

                {tabValue === "base_dist" && (
                <Container className="mt-3">
                    {Object.entries(base_dist).map(([rowKey, columns]) => (
                        <div key={rowKey}>
                            <Row>
                                {Object.entries(columns).map(([field, value]) => {
                                    // Logic: Disable if it's a 'mirrored' field (yx, zx, zy)
                                    // This forces the user to use xy, xz, and yz instead.
                                    const isMirrored = ["yx", "zx", "zy"].includes(field);
                                    
                                    return (
                                        <Col md={4} key={field}>
                                            <Form.Group controlId={field}>
                                                <Form.Label>{field}</Form.Label>
                                                <Form.Control
                                                    type="number"
                                                    step="0.01"
                                                    disabled={isMirrored} // Keep the UI symmetric but locked
                                                    {...register(`base_dist.${rowKey}.${field}`)}
                                                />
                                            </Form.Group>
                                        </Col>
                                    );
                                })}
                            </Row>
                        </div>
                    ))}
                </Container>
                )}

                {tabValue === "import" && (
                    <Container className="mt-3">
                        <Row><h2>Work in Progress, please contact repository owner!</h2></Row>
                        <Row><h5>This tab will use a normal Gaussian distribution</h5></Row>
                    </Container>
                )}

                {/* Submit */}
                <div className="d-flex justify-content-center mt-4">
                <Button type="submit" variant="light">
                    Save
                </Button>
                </div>

            </Container>
        </Form>
    )
};

export default ParticleSettings;
 
// Optional Enhancements (Highly Recommended)

// Disable Simulate unless isDirty

// Warn on modal close if isDirty

// Add schema validation (Zod/Yup)

// Persist last-used values via reset()