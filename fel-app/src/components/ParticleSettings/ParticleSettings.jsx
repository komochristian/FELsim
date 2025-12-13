import { Container, Row, Button, Col, Form } from "react-bootstrap";
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import { useState } from "react";
import { useForm } from "react-hook-form";


const initialInputs = {
    x: { α: '', β: '', φ: '', ε: '' },
    y: { α: '', β: '', φ: '', ε: '' },
    z: { α: '', β: '', φ: '', ε: '' },
  };


const ParticleSettings = ({ setSelectedMenu, setBeamInput, currentBeamType, setBeamtypeToPass, beamtypeToPass,
    numOfParticles, setParticleNum, beamlistSelected, getBeamline, mev, setMeV }) => {
    const [tabValue, setTabValue] =  useState('one');
    const { register, handleSubmit } = useForm({
        defaultValues: initialInputs,
      });
    
    //  TO DO: Add a schema validation with yup 
    const onSubmit = (data) => {
        console.log("Form values:", data);
        setSelectedMenu(null);
        getBeamline(beamlistSelected);
    };
    const axes = ["x", "y", "z"];
    const letters = ["α", "β", "φ", "ε"];

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
        <Container className="d-flex flex-column align-items-center justify-content-center">
            <h4>Particle Settings</h4>
            <label htmlFor="beamtypeSelect" className="forLabels">Select Beam type:</label>
            <select name="beamtypeSelect" 
                    onChange={(e) => setBeamInput(e.target.value)}
                    value={currentBeamType}>
                <option value="electron">Electron</option>
                <option value="proton">Proton</option>
                <option value="otherIon">Other Ion</option>
            </select>
            {
                (currentBeamType !== "electron" && currentBeamType !== "proton") && (
                <input
                    type="text"
                    onChange={(e) => setBeamtypeToPass(e.target.value)}
                    value={beamtypeToPass}
                />)
            }
            <label htmlFor="numParticles" className="forLabels">Number of particles:</label>
            <input defaultValue={numOfParticles}
                    type="number"
                    name="numParticles" 
                    onChange={(e) => setParticleNum(e.target.value)}
                    min={3}
            />
            <label htmlFor="kineticEnergy" className="forLabels">Kinetic Energy (MeV):</label>
            <input defaultValue={mev}
                    type="number"
                    name="kineticEnergy" 
                    onChange={(e) => setMeV(e.target.value)}
                    min={0}
            />
            <Box>
                <Tabs
                    value={tabValue}
                    onChange={handleChange}
                    centered
                >
                    <Tab value="one" label="Twiss Parameters" />
                    <Tab value="two" label="Base Distributions" />
                    <Tab value="three" label="Import" />
                </Tabs>
            </Box>
            {tabValue === "one" && 
                <Container>
                    <Form onSubmit={handleSubmit(onSubmit)}>
                    {axes.map((axis) => (
                        <div key={axis} className="mb-4">
                        <Row>
                            {letters.map((letter) => (
                            <Col md={3} key={letter}>
                                <Form.Group controlId={`${axis}-${letter}`}>
                                <Form.Label>{axis}: {letter}</Form.Label>
                                <Form.Control
                                    type="text"
                                    {...register(`${axis}.${letter}`)}
                                />
                                </Form.Group>
                            </Col>
                            ))}
                        </Row>
                        </div>
                    ))}
                    <div className="d-flex justify-content-center">
                        <Button type="submit" variant="light">
                            Simulate
                        </Button>
                    </div>
                    </Form>
                </Container>
            }
        </Container>
    )
};

export default ParticleSettings;