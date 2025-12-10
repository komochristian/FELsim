import ExcelUploadButton from "../ExcelUploadButton/ExcelUploadButton";
import './BeamSettings.css';
import { Container, Row, Button } from "react-bootstrap";

const BeamSettings = ({ setSelectedMenu, excelToAPI, sInterval, setSInterval,
    beamlistSelected, getBeamline }) => {
    return (
        <Container>
            <h4>Graph Settings</h4>
            <ExcelUploadButton excelToAPI={excelToAPI} />
            <label htmlFor="interval" className="forLabels">S axis interval</label>
            <input defaultValue={sInterval}
                    type="number"
                    name="interval" 
                    onChange={(e) => setSInterval(e.target.value)}
            />
            <Row className="mt-2">
                <Button
                    variant="light"
                    onClick={() => {
                        setSelectedMenu(null);
                        getBeamline(beamlistSelected);
                    }}
                    bold
                >
                    Simulate
                </Button>
            </Row>
        </Container>
    )
};

export default BeamSettings;