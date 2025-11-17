import React, {useState, useEffect} from 'react';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import ExcelUploadButton from './components/ExcelUploadButton/ExcelUploadButton';
import LineGraph from './components/LineGraph/LineGraph';
import ErrorWindow from './components/ErrorWindow/ErrorWindow';
import Select from 'react-select';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import FloatingInfoButton from './components/FloatingInfoButton/FloatingInfoButton';
import { Table } from 'rsuite';
const { Column, HeaderCell } = Table;
import 'rsuite/dist/rsuite.min.css'; 
import ActionCell from './components/ActionCell/ActionCell';
import EditableCell from './components/EditableCell/EditableCell';
import NormalCell from './components/NormalCell/NormalCell';
import { Col, Row } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Modal } from 'react-responsive-modal';
import 'react-responsive-modal/styles.css';
import ModalContent from './components/ModalContent/ModalContent';
import { PRIVATEVARS, API_ROUTE } from './constants';

function App()
{
    console.log(API_ROUTE);
    const [beamSegmentInfo, setData] = useState(null);
    const [dotGraphs, setDotGraphs] = useState([]);
    const [lineGraph, setLineGraph] = useState(null);
    const [beamlistSelected, setSelectedItems] = useState([]);
    const [currentZ, setZValue] = useState(0);
    const [currentBeamType, setBeamInput] = useState('electron');
    const [beamtypeToPass, setBeamtypeToPass] = useState('electron');
    const [twissDf, setTwissDf] = useState([]);
    const [totalLen, setTotalLen] = useState(0);
    const [numOfParticles, setParticleNum] = useState(1000);
    const [zInterval, setZInterval] = useState(0.1);
    const [showError, setError] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [scroll, setScroll] = useState(false); // State for the checkbox
    const [twissOptions, setTwissOptions] = useState([
                                    { value: '\\epsilon (\\pi.mm.mrad)', label: '\\epsilon (\\pi.mm.mrad)' },
                                    { value: '\\alpha', label: '\\alpha' },
                                    { value: '\\beta (m)', label: '\\beta (m)' },
                                    { value: '\\gamma (rad/m)', label: '\\gamma (rad/m)' },
                                    { value: 'D (mm)', label: 'D (mm)' },
                                    { value: 'D\' (mrad)', label: 'D\' (mrad)' },
                                    { value: '\\phi (deg)', label: '\\phi (deg)' },
                                    { value: 'Envelope\\ E (mm)', label: 'Envelope\\ E (mm)' }
                                ]);
    const [currentTwissParam, setCurrentTwiss] = useState({value: 'Envelope\\ E (mm)',
                                                           label: 'Envelope\\ E (mm)'});
    const [selectedMenu, setSelectedMenu] = useState(null);

    const showErrorWindow = (message) => {
        console.log("Error:", message);
        setErrorMessage(message);
        setError(true);
    };
    
    const errorCatcher = () => {
        if (zInterval <= 0) {
            showErrorWindow("Please use an interval value greater than 0");
            return true;
        };
        if (numOfParticles < 3) {
            showErrorWindow("Use at least 3 particles");
            return true;
        };
        if (beamlistSelected.length == 0) {
            showErrorWindow("Please include 1+ beam elements");
            return true;
        } 
        return false;         
    };

    useEffect(() => {
        if (!showError) return ;
        const timer = setTimeout(() => setError(false), 4000);
        return () => clearTimeout(timer); 
    }, [showError]);

    useEffect(() => {
        if (currentBeamType === "proton") {
            setBeamtypeToPass(() => "proton")
        }
        else if ( currentBeamType === "electron") {
            setBeamtypeToPass(() => "electron")
        }
    }, [currentBeamType]);

    useEffect(() => {
        setZValue(() => 0);
    }, [dotGraphs]);     

    useEffect(() => {
        fetch(API_ROUTE + '/beamsegmentinfo')
            .then((response) => response.json())
            .then((json) => setData(json))
            .catch((err) => console.error("Error loading beam segment info:", err));
        }, []);
    //console.log(beamSegmentInfo);

    useEffect(() => {
        console.log("Updated beamlistSelected:", beamlistSelected);
    }, [beamlistSelected]);

    if (!beamSegmentInfo) return <div>Loading...</div>;
    const items = Object.keys(beamSegmentInfo);

    //  Calculates the start and end position of the entire beamline,
    //  Assumes segment format is already correct
    const beamlistHandler = (segList) => {
        let zCurrent = 0;
        const cleanedSegList = segList.map((obj, i) => {
            obj['startPos'] = zCurrent;
            zCurrent += obj['length'];
            obj['endPos'] = zCurrent;
            obj.id = i;
            return obj;
        })
        setTotalLen(zCurrent);
        setSelectedItems(cleanedSegList);
    };


    //  Handles the color of a single segment, use if no need to color an
    //  entire beamline
    const handleSegmentColor = (segment) => { 
        for (let priv of PRIVATEVARS) {
            if (!(priv in segment)) {
                const segName = segment.name || Object.keys(segment)[0];
                segment[priv] = beamSegmentInfo[segName][priv];
            }
        }
        return segment
    };

    //  Handles both color and start and end pos for ENTIRE beamline
    const setSelectedItemsHandler = (segList) => {
        console.log("segList from excel:", segList);
        const cleanedSegList = segList.map((segment) => {
            const name = Object.keys(segment)[0];             
            return handleSegmentColor({ "name": name,
                                        ...segment[name]});
        });
        beamlistHandler(cleanedSegList);
    };
    
    //  Handles and formats twiss data for plotting
    const handleTwiss = (twissJsonObj, x_axis) => { 
        //console.log(twissJsonObj);
        const twissPlotData = Object.entries(twissJsonObj).flatMap(([key, obj]) => {
                return Object.entries(obj).map(([axis, arr], index) => { 
                    return {
                            "id": `${key}: ${axis}`,
                            "data": 
                                arr.map((val, i) => ({
                                    'x': x_axis[i],
                                    'y': val
                                    })
                                ) 
                    } 
                });
        });
       
        const grouped = twissPlotData.reduce((acc, item, i) => {
          const label = twissOptions[Math.floor(i / 3)].label;
        
          // Check if label group already exists
          //let group = acc.find(g => g.label === label);
          let group = acc[label];
          if (!group) {
              acc[label] = [];
          }
          
          acc[label].push(item);
          return acc;
        }, []);

        // console.log(grouped);
        //console.log(twissPlotData);
        //setTwissDf(twissPlotData);
        setTwissDf(grouped);
    };

    const excelToAPI = async (fileJSON) => {  
        const res =  await fetch(API_ROUTE + '/excel-to-beamline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(fileJSON, null, 2),
        });
        if (!res.ok) {
            const errorData = await res.json();
            showErrorWindow(`Bad excel file input format, from server: ${errorData.detail || errorData}`);
            return 
        }
        const beamlist =  await res.json();
        setSelectedItemsHandler(beamlist);
    };
    //CHANGE
    const handleItemClick = (item) => {
        const beamObj = handleSegmentColor({[item]: structuredClone(beamSegmentInfo[item])});

        const cleanedObj = {"name": item,
                            ...beamObj[item]};
        console.log('cleanedObj', cleanedObj);
        //console.log('updated Obj', beamObj);
        const updatedList = [...beamlistSelected, cleanedObj];
        beamlistHandler(updatedList);
    };

    //CHANGE
    const getBeamline = async (segList) => {
        const uiErrorStatus = errorCatcher();
        if (uiErrorStatus) {
            return
        };
        const cleanedList = segList.map(obj => {
            const key = obj.name;
            const cleanedParams = Object.fromEntries(
              Object.entries(obj).filter(([p]) => !PRIVATEVARS.includes(p))
            );
            console.log('cleanedParams:', cleanedParams);
            return {
                segmentName: key,
                parameters: cleanedParams
            };
        });

        const plottingParams = {
            beamlineData: cleanedList,
            num_particles: numOfParticles,
            beamType: beamtypeToPass,
            interval: zInterval
        }
    
        const jsonBody = JSON.stringify(plottingParams, null, 2); 
        //console.log("json sent;", jsonBody);

        const res = await fetch(API_ROUTE + '/axes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: jsonBody,
        });
        if (!res.ok) {
            const errorData = await res.json();
            showErrorWindow(errorData.detail || errorData);
            return 
        }
        const axImages = await res.json();
        const result = axImages['images'];
        const lineAxObj = axImages['line_graph'];
        const lineAx = lineAxObj['axis'];

        handleTwiss(JSON.parse(lineAxObj['twiss']), lineAxObj['x_axis']);

        const cleanResult = new Map(
            Object.entries(result).map(([key, value]) => [
                parseFloat(key),
                `data:image/png;base64,${value}`,
              ])
            );

        setDotGraphs(cleanResult);
        setLineGraph(`data:image/png;base64,${lineAx}`);
        //console.log("returned api result:", result);
        //console.log("newSubArr:", cleanResult);

    };

    const handleChange = (id, key, value) => {
        const nextData = Object.assign([], beamlistSelected);
        nextData.find(item => item.id === id)[key] = value;
      };

    const handleEdit = id => {
        const nextData = Object.assign([], beamlistSelected);
        const activeItem = nextData.find(item => item.id === id);

        if(activeItem.status === 'EDIT') {
            const newItem = beamlistSelected.find(item => item.id === id);
            if (newItem) {
                const { status, ...rest } = newItem;
                Object.assign(activeItem, rest);
            }
        }
        activeItem.status = activeItem.status === 'EDIT' ? null : 'EDIT';
        beamlistHandler(nextData);
      };
    
      const handleRemove = id => {
        const beamlineHandler = beamlistSelected.filter(item => item.id !== id);
        beamlistHandler(beamlineHandler);
      };

    const PreModelCheck = (beamline) => {
        if (beamline.length === 0) {
            setSelectedMenu(null)
            showErrorWindow("Please add beam segments before graphing parameters");
            return false;
        }
        return true;
    }
      
    return (
        <>
        <ErrorWindow message={errorMessage}
                     showError = {showError} />
        <Modal open={selectedMenu === 'parameterGraphing' && PreModelCheck(beamlistSelected)} 
               onClose={() => setSelectedMenu(null)} 
               center
               classNames={{
                modal: "custom-modal", // Add a custom class to the modal
              }} 
        >
            <div className="modal-content">
                {/* CHANGE */}
                <ModalContent beamline={beamlistSelected} />
            </div>
        </Modal> 
        <div className="layout">
        <FloatingInfoButton /> 
          <div className={`sidebar ${selectedMenu === null ? 'menuClosed' : 'menuOpen'}`}>
            <h2>FEL simulator</h2>
            <div>
                <Dropdown buttonText="Add Segment" 
                        contentText={
                                <>
                                    {items.map((item) => (
                                        <DropdownItem key={item}
                                                    onClick={() => handleItemClick(item)}
                                        >
                                            {`${item}`}
                                        </DropdownItem>
                                    ))}
                                </>
                        }
                />
                <button
                    type="button"
                    className="simButton"
                    onClick={() => getBeamline(beamlistSelected)}>
                    Simulate
                </button>
            </div>
            <h4>Beam setup</h4>
            <div className="scrollBox">
                    {/* // CHANGE */}
                    {/* ALLOW EDITTING OF ALL PARAMETERS LATER ON */}
                    <Table height={420} data={beamlistSelected}>
                        <Column flexGrow={1}>
                            <HeaderCell>Name</HeaderCell>
                            <NormalCell dataKey="name" />
                        </Column>

                        <Column flexGrow={1}>
                            <HeaderCell>length</HeaderCell>
                            <EditableCell
                                dataKey="length"
                                dataType="number"
                                onChange={handleChange}
                            />
                        </Column>

                        {selectedMenu !== "beamSettings" &&
                        <>
                            <Column flexGrow={1}>
                                <HeaderCell>angle</HeaderCell>
                                <EditableCell
                                    dataKey="angle"
                                    dataType="number"
                                    onChange={handleChange}
                                />
                            </Column>

                            <Column flexGrow={1}>
                                <HeaderCell>current</HeaderCell>
                                <EditableCell
                                    dataKey="current"
                                    dataType="number"
                                    onChange={handleChange}
                                />
                            </Column> 
                        </>
                        }

                        <Column width={100}>
                            <HeaderCell>Action</HeaderCell>
                            <ActionCell dataKey="id" onEdit={handleEdit} onRemove={handleRemove} />
                        </Column>
                    </Table>
            </div>
          </div>
          { selectedMenu === 'beamSettings' ?
          <>
            <div className="beamSettings">
            <button className="close-button" onClick={() => setSelectedMenu(null)}>
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
                <label htmlFor="interval" className="forLabels">Z axis interval</label>
                <input defaultValue={zInterval}
                    type="number"
                        name="interval" 
                        onChange={(e) => setZInterval(e.target.value)}
                />
            </div>
          <div className="toggleLegend">
            <label>
                <input
                    type="checkbox"
                    checked={scroll}
                    onChange={(e) => setScroll(e.target.checked)} // Update scroll state
                />
                Enable Scroll
            </label>
            <Select className='select-container'
                    options={twissOptions}
                    value={currentTwissParam}
                    onChange={setCurrentTwiss}
                    getOptionLabel={e => <InlineMath math={e.label} />}
                    getSingleValueLabel={e => <InlineMath math={e.label} />}
                    />
          </div>
        </>
        :
        <div className='menu-options'>
            <Col className="settings-icon-wrapper pt-3">
                <Row className="mb-3 g-0">
                    <button className="menu-button" onClick={() => setSelectedMenu("beamSettings")}>
                        <i className="fas fa-cog"></i>
                    </button>
                </Row>
                <Row className="mb-3 g-0">
                    <button className="menu-button" onClick={() => setSelectedMenu("parameterGraphing")}>
                        <i className="fas fa-chart-line"></i>
                    </button>
                </Row>   
            </Col>
        </div>
        }  
          <div className={`main-content`}>
                <img src={dotGraphs.size > 0 ? dotGraphs.get(currentZ) : null} alt="Please run simulation"/>
          </div>
          <div className="twiss-graph">
                <LineGraph twissData={twissDf}
                           setZValue={setZValue} 
                           beamline={beamlistSelected}
                           totalLen={totalLen}
                           twissAxis={currentTwissParam}
                           scroll={scroll}
                           setScroll={setScroll}>
                </LineGraph>
            {/*<img src={lineGraph ? lineGraph: null} alt="loading..."/>*/}
          </div>
          
        </div>
        </>
    );
}

export default App;
