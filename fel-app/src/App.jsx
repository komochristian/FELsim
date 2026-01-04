import React, {useState, useEffect, useRef} from 'react';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
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
import { Col, Row, Card, Overlay, Button } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Modal } from 'react-responsive-modal';
import 'react-responsive-modal/styles.css';
import ModalContent from './components/ModalContent/ModalContent';
import { PRIVATEVARS, API_ROUTE, TWISS_OPTIONS } from './constants';
import { Mosaic } from 'react-loading-indicators';
import BeamSettings from './components/BeamSettings/BeamSettings';
import ParticleSettings from './components/ParticleSettings/ParticleSettings';
import PlotMenu from './components/PlotMenu/PlotMenu';
import GoToSPosition from './components/GoToSPosition/GoToSPosition';
import SimulationModel from './components/SimulationModel/SimulationModel';

function App()
{
    console.log(API_ROUTE);
    const [beamSegmentInfo, setBeamSegmentInfo] = useState(null);
    const [dotGraphs, setDotGraphs] = useState([]);
    const [lineGraph, setLineGraph] = useState(null);
    const [beamlistSelected, setSelectedItems] = useState([]);
    const [currentS, setSValue] = useState(0);
    const [beamtypeToPass, setBeamtypeToPass] = useState('electron');
    const [twissDf, setTwissDf] = useState([]);
    const [totalLen, setTotalLen] = useState(0);
    const [numOfParticles, setParticleNum] = useState(1000);
    const [sInterval, setSInterval] = useState(0.1);
    const [showError, setError] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [scroll, setScroll] = useState(false); // State for the checkbox
    const [currentTwissParam, setCurrentTwiss] = useState({value: 'Envelope\\ E (mm)',
                                                           label: 'Envelope\\ E (mm)',
                                                           modal_val: 'envelope'});
    const [selectedMenu, setSelectedMenu] = useState(null);
    const [selectedRowId, setSelectedRowId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [mev, setMeV] = useState(45);
    const [twissValues, setTwissValues] = useState({
        x: { alpha: '0', beta: '1', phi: '0', epsilon: '1' },
        y: { alpha: '0', beta: '1', phi: '0', epsilon: '1' },
        z: { alpha: '0', beta: '1', phi: '0', epsilon: '10' },
    });
    const [base_distribution, setBaseDistribution] = useState(
        {
            sigma_x: {x: 0, y: 0, z: 0},
            sigma_y: {x: 0, y: 0, z: 0},
        }
    );
    const [beamSetup, setBeamSetup] = useState("base_dist");
    const [showGraphSettings, setShowGraphSettings] = useState(false);
    const [graphTarget, setTarget] = useState(null);

    const showErrorWindow = (message) => {
        console.log("Error:", message);
        setErrorMessage(message);
        setError(true);
    };

    const errorCatcher = () => {
        if (sInterval <= 0) {
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
        console.log(beamlistSelected);
    }, [beamlistSelected]);

    useEffect(() => {
        if (!showError) return ;
        const timer = setTimeout(() => {
            setError(false);
            
            const secondTimer = setTimeout(() => {
                setErrorMessage('');
            }, 300);

            // cleanup inner timer
            return () => clearTimeout(secondTimer);
        }, 4000);
        return () => clearTimeout(timer); 
    }, [showError]);

    useEffect(() => {
        setSValue(() => 0);
    }, [dotGraphs]);     

    useEffect(() => {
        fetch(API_ROUTE + '/beamsegmentinfo')
            .then((response) => response.json())
            .then((json) => setBeamSegmentInfo(json))
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
        let sCurrent = 0;
        const cleanedSegList = segList.map((obj, i) => {
            obj['startPos'] = sCurrent;
            sCurrent += obj['length'];
            obj['endPos'] = sCurrent;
            obj.id = i;
            return obj;
        })
        setTotalLen(sCurrent);
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
          const label = TWISS_OPTIONS[Math.floor(i / 3)].label;
        
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
        const insertIndice = selectedRowId !== null ? selectedRowId : beamlistSelected.length;
        const updatedList = [...beamlistSelected.slice(0, insertIndice), cleanedObj, ...beamlistSelected.slice(insertIndice)];
        setSelectedRowId((id) => id === null ? null : id + 1);
        beamlistHandler(updatedList);
    };

    //CHANGE
    const getBeamline = async (segList) => {
        if (loading) {
            showErrorWindow("Simulation already in progress");
            return;
        };
        setLoading(true);
        try {
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
                interval: sInterval,
                kineticE: mev,
                beam_setup: beamSetup,
                twiss: twissValues,
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
            
            //  Must use map to maintain key consistency
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
        } finally {
            setLoading(false);
        }
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

    const PreModalCheck = (beamline) => {
        if (beamline.length === 0) {
            setSelectedMenu(null)
            showErrorWindow("Please add beam segments before graphing parameters");
            return false;
        }
        return true;
    }

    const ParticleSettingsSubmitHelper = (data) => {
        console.log("Settings data received:", data);
        setBeamtypeToPass(data.customIon ? data.customIon : data.beamType);
        setMeV(data.kineticEnergy);
        setParticleNum(data.numParticles);
        setBaseDistribution(data.base_distribution);
        setTwissValues(data.twiss);
        if (data.beam_setup === "twiss") {
            setBeamSetup("twiss");
        }
        else if (data.beam_setup === "base_dist") {
            setBeamSetup("base_dist");
        }
        else if (data.beam_setup === "import") {
            setBeamSetup("import");
        }
    };

    const SaveFig = () => {
        if (!dotGraphs || dotGraphs.length === 0 || !dotGraphs.get(currentS) || dotGraphs.size === 0) {
            showErrorWindow("No simulation loaded to save");
            return;
        }
        const link = document.createElement('a');
        link.href = dotGraphs.get(currentS);
        link.download = `beam_plot_${currentS}.png`; // or any filename
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const goToSPos = (data) => {
        setSelectedMenu(null);
        if (!dotGraphs || dotGraphs.size === 0 || dotGraphs.length === 0) { showErrorWindow("No simulation loaded"); return; }
        const target = data.s_pos;
        if (dotGraphs.get(target)) {
             setSValue(target);
             return;
        };

        const arr = Array.from(dotGraphs.keys());
        const closestS = arr.reduce((prev, curr) =>
            Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
        );
        setSValue(closestS);
    };
      
    return (
        <>
        <ErrorWindow message={errorMessage}
                     showError = {showError} />
        <Modal 
            open={selectedMenu === 'parameterGraphing' && PreModalCheck(beamlistSelected)} 
            onClose={() => setSelectedMenu(null)} 
            center
            classNames={{
                modal: "parameter-modal", // Add a custom class to the modal
            }} 
        >
            <div className="modal-content">
                <ModalContent beamline={beamlistSelected} showErrorWindow={showErrorWindow} />
            </div>
        </Modal>
        <Modal 
            open={selectedMenu === 'beamSettings'} 
            onClose={() => setSelectedMenu(null)} 
            center
            classNames={{
                modal: "setting-modal", // Add a custom class to the modal
            }} 
        >
            <div className="beamSettings">
                <BeamSettings
                    setSelectedMenu={setSelectedMenu}
                    excelToAPI={excelToAPI}
                    sInterval={sInterval}
                    setSInterval={setSInterval}
                    beamlistSelected={beamlistSelected}
                    getBeamline={getBeamline}
                />
         </div>
        </Modal>  
        <Modal 
            open={selectedMenu === 'particleSettings'} 
            onClose={() => setSelectedMenu(null)} 
            center
            classNames={{
                modal: "setting-modal", // Add a custom class to the modal
            }} 
        >
            <ParticleSettings
                setSelectedMenu={setSelectedMenu}
                beamtypeToPass={beamtypeToPass}
                numOfParticles={numOfParticles}
                mev={mev}
                submitHelper={ParticleSettingsSubmitHelper}
                twissValues={twissValues}
                base_distribution={base_distribution}
                beamSetup={beamSetup}
            />
        </Modal>
        <Modal
            open={selectedMenu === 'simulationModel'} 
            onClose={() => setSelectedMenu(null)} 
            center
        >
            <SimulationModel></SimulationModel>
        </Modal>
        <Modal
            open={selectedMenu === 's_pos_select'}
            onClose={() => setSelectedMenu(null)}
            center
        >
            <GoToSPosition
                goToSPos={goToSPos}
            />
        </Modal>
        <div className="layout">
        <FloatingInfoButton /> 
        <div className={`sidebar`}>
            <h2>FEL simulator</h2>
            <Row>
                <Col>
                    <Dropdown 
                        buttonText="Add Segment" 
                        contentText={
                            <>
                                {items.map((item) => (
                                    <DropdownItem 
                                        key={item}
                                        onClick={() => handleItemClick(item)}
                                    >
                                        {`${item}`}
                                    </DropdownItem>
                                ))}
                            </>
                        }
                    />
                    <Button
                        onClick={() => getBeamline(beamlistSelected)}
                        variant="light"
                        className="rounded-0 fw-bold border border-dark"
                    >
                        Simulate
                    </Button>
                </Col>
            </Row>
            <h4>Beam setup</h4>
            <div className="scrollBox">
                {/* ALLOW EDITTING OF ALL PARAMETERS LATER ON */}
                <Table height={420} 
                       data={beamlistSelected}
                       onRowClick={(rowData) => {
                            // rowData contains the clicked row's info
                            if (rowData.id === selectedRowId) {
                                setSelectedRowId(null);
                            } else 
                            setSelectedRowId(rowData.id)
                        }}
                        rowClassName={rowData =>
                            `${rowData?.id === selectedRowId ? 'highlight-row' : ''} table-hover-row`
                        }
                        >
                    <Column flexGrow={1} fullText>
                        <HeaderCell>Name</HeaderCell>
                        <NormalCell dataKey="name" />
                    </Column>

                    <Column flexGrow={1} fullText>
                        <HeaderCell>length</HeaderCell>
                        <EditableCell
                            dataKey="length"
                            dataType="number"
                            onChange={handleChange}
                        />
                    </Column>
                    <Column flexGrow={1} fullText>
                        <HeaderCell>angle</HeaderCell>
                        <EditableCell
                            dataKey="angle"
                            dataType="number"
                            onChange={handleChange}
                        />
                    </Column>
                    <Column flexGrow={1} fullText>
                        <HeaderCell>current</HeaderCell>
                        <EditableCell
                            dataKey="current"
                            dataType="number"
                            onChange={handleChange}
                        />
                    </Column>
                    <Column width={100}>
                        <HeaderCell>Action</HeaderCell>
                        <ActionCell dataKey="id" onEdit={handleEdit} onRemove={handleRemove} />
                    </Column>
                </Table>
            </div>
          </div>
        <div className='menu-options'>
            <Col className="settings-icon-wrapper pt-3 h-100 d-flex flex-column">
                <Row className="mb-3 g-0">
                    <button 
                        className="menu-button" 
                        onClick={() => {
                            setSelectedMenu("beamSettings");
                        }}
                    >
                        <i className="fas fa-cog"></i>
                    </button>
                </Row>
                <Row className="mb-3 g-0">
                    <button className="menu-button" onClick={() => setSelectedMenu("parameterGraphing")}>
                        <i className="fa-solid fa-chart-area"></i>
                    </button>
                </Row>
                <Row className="mb-3 g-0">
                    <button 
                        className="menu-button" 
                        onClick={() => {
                            setSelectedMenu("particleSettings");
                            }}
                        >
                        <i className="fas fa-atom"></i>
                    </button>
                </Row> 
                <Row className="mb-3 g-0">
                    <button 
                        className="menu-button" 
                        onClick={() => {
                            setSelectedMenu("simulationModel");
                            }}
                    >
                       <i class="fa-solid fa-network-wired"></i>
                    </button>
                </Row> 
                <Row className="g-0 mt-auto mb-3">
                    <button 
                        className="menu-button"
                        onClick={(e) => {
                            setShowGraphSettings(!showGraphSettings);
                            setTarget(e.target);
                        }}
                    >
                        <i className="fas fa-chart-line"></i>
                    </button>
                </Row>
            </Col>
            <Overlay
                show={showGraphSettings}
                target={graphTarget}
                placement="right"
                containerPadding={20}
                popperConfig={{
                    modifiers: [
                      {
                        name: "offset",
                        options: {
                          offset: [0, 16], // [skid, distance]
                        },
                      },
                    ],
                  }}
                rootClose
                onHide={() => setShowGraphSettings(false)}
            >
            {(props) => (
                <Card {...props} className="mt-3">
                <Card.Header>
                    Graph settings
                </Card.Header>
                <label>
                    <input
                        type="checkbox"
                        checked={scroll}
                        onChange={(e) => setScroll(e.target.checked)} // Update scroll state
                    />
                    Enable Scroll
                </label>
                    <Select className='select-container'
                            options={TWISS_OPTIONS}
                            value={currentTwissParam}
                            onChange={setCurrentTwiss}
                            getOptionLabel={e => <InlineMath math={e.label} />}
                            getSingleValueLabel={e => <InlineMath math={e.label} />}
                            menuPortalTarget={document.body}
                            menuPosition="fixed"
                            styles={{
                                menuPortal: (base) => ({ ...base, zIndex: 9999 })
                            }}
                            />
                </Card>
            )}
            </Overlay>
        </div>
        <div className="main-content h-100 d-flex justify-content-center align-items-center">
            <PlotMenu 
                saveFig={SaveFig}
                setSelectedMenu={setSelectedMenu}
            />
            {loading ? <Mosaic color="#000000" size="small" text="Loading" textColor="#000000" />
            :
            (dotGraphs.size > 0 ? <img src={dotGraphs.get(currentS)}/> : <h1>No simulation loaded</h1>)
            }
        </div>
        <div className="twiss-graph">
            <LineGraph 
            twissData={twissDf}
            setSValue={setSValue} 
            beamline={beamlistSelected}
            totalLen={totalLen}
            twissAxis={currentTwissParam}
            scroll={scroll}
            setScroll={setScroll} />
        </div>
        </div>
        </>
    );
}

export default App;
