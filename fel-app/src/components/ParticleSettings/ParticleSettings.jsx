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
    // const [inputs, setInputs] = useState(initialInputs);
    const { register, handleSubmit } = useForm({
        defaultValues: initialInputs,
      });
    
      const onSubmit = (data) => {
        console.log("Form values:", data);
      };
      const axes = ["x", "y", "z"];
      const letters = ["α", "β", "φ", "ε"];

    // // Generic handler
    // const handleInputChange = (plane, field, value) => {
    //   setInputs(prev => ({
    //     ...prev,
    //     [plane]: {
    //       ...prev[plane],
    //       [field]: value,
    //     },
    //   }));
    // };

    const handleChange = (event, newValue) => {
        setTabValue(newValue);
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

            <Container>
                {/* {['x', 'y', 'z'].map(plane => (
                    <div key={plane}>
                    <h5>{plane.toUpperCase()}</h5>
                    {['α', 'β', 'φ', 'ε'].map(field => (
                        <Col xs={3} key={field}>
                            <input
                            key={field}
                            type="number"
                            value={inputs[plane][field]}
                            onChange={e => handleInputChange(plane, field, e.target.value)}
                            // placeholder={`${plane} ${field}`}
                            />
                        </Col>
                    ))}
                    </div>
                ))} */}
                <Form onSubmit={handleSubmit(onSubmit)}>
                {axes.map((axis) => (
                    <div key={axis} className="mb-4">
                    <h5>{axis.toUpperCase()}</h5>

                    <Row>
                        {letters.map((letter) => (
                        <Col md={3} key={letter}>
                            <Form.Group controlId={`${axis}-${letter}`}>
                            <Form.Label>{axis}.{letter}</Form.Label>
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

                <Button type="submit" variant="primary">
                    Submit
                </Button>
                </Form>
            </Container>
            <Row className="mt-2">
                <Button
                    variant="light"
                    onClick={() => {
                        setSelectedMenu(null);
                        getBeamline(beamlistSelected);
                    }}
                >
                    Simulate
                </Button>
            </Row>
        </Container>
    )
};

export default ParticleSettings;

// PEAKGPT CODE PROVIDED
// import { useForm } from "react-hook-form";
// import Form from "react-bootstrap/Form";
// import Row from "react-bootstrap/Row";
// import Col from "react-bootstrap/Col";
// import Button from "react-bootstrap/Button";

// const initialInputs = {
//   x: { α: "", β: "", φ: "", ε: "" },
//   y: { α: "", β: "", φ: "", ε: "" },
//   z: { α: "", β: "", φ: "", ε: "" },
// };

// export default function ExampleForm() {
//   const { register, handleSubmit } = useForm({
//     defaultValues: initialInputs,
//   });

//   const onSubmit = (data) => {
//     console.log("Form values:", data);
//   };

//   const axes = ["x", "y", "z"];
//   const letters = ["α", "β", "φ", "ε"];

//   return (
//     <Form onSubmit={handleSubmit(onSubmit)}>
//       {axes.map((axis) => (
//         <div key={axis} className="mb-4">
//           <h5>{axis.toUpperCase()}</h5>

//           <Row>
//             {letters.map((letter) => (
//               <Col md={3} key={letter}>
//                 <Form.Group controlId={`${axis}-${letter}`}>
//                   <Form.Label>{axis}.{letter}</Form.Label>
//                   <Form.Control
//                     type="text"
//                     {...register(`${axis}.${letter}`)}
//                   />
//                 </Form.Group>
//               </Col>
//             ))}
//           </Row>
//         </div>
//       ))}

//       <Button type="submit" variant="primary">
//         Submit
//       </Button>
//     </Form>
//   );
// }
