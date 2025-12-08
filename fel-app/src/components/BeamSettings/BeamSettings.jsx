import ExcelUploadButton from "../ExcelUploadButton/ExcelUploadButton";
import './BeamSettings.css';

const BeamSettings = ({ setSelectedMenu, setSidebarOpen, excelToAPI, setBeamInput, currentBeamType, setBeamtypeToPass, beamtypeToPass,
    numOfParticles, setParticleNum, sInterval, setSInterval }) => {
    return (
        <>
        <h4>Simulation Settings</h4>
                <button 
                    className="close-button" 
                    onClick={() => {
                        setSelectedMenu(null);
                        setSidebarOpen(false)
                }}
                >
                    X
                </button>
                <ExcelUploadButton excelToAPI={excelToAPI} />
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
                <label htmlFor="interval" className="forLabels">S axis interval</label>
                <input defaultValue={sInterval}
                        type="number"
                        name="interval" 
                        onChange={(e) => setSInterval(e.target.value)}
                />
        </>
    )
};

export default BeamSettings;