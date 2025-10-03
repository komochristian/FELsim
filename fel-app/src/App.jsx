import React, {useState, useEffect} from 'react';
import * as d3 from 'd3';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import BeamSegment from './components/BeamSegment/BeamSegment';
import ExcelUploadButton from './components/ExcelUploadButton/ExcelUploadButton';
import LineGraph from './components/LineGraph/LineGraph';
import ErrorWindow from './components/ErrorWindow/ErrorWindow';
import Select from 'react-select';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

function App()
{
    const PRIVATEVARS = ['color', 'startPos', 'endPos'];  // USE THIS SO USERS CANT EDIT THESE VALUES
    const API_ROUTE = import.meta.env.VITE_DOCKER_ROUTE || 'http://127.0.0.1:8000';
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
    
    const showErrorWindow = (message) => {
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
        //console.log("Updated beamlistSelected:", beamlistSelected);
    }, [beamlistSelected]);

    if (!beamSegmentInfo) return <div>Loading...</div>;
    const items = Object.keys(beamSegmentInfo);

    //  Calculates the start and end position of the entire beamline
    const calcStartEndPos = (segList) => {
        let zCurrent = 0;
        const cleanedSegList = segList.map((obj) => {
            const segName = Object.keys(obj)[0];
            obj[segName]['startPos'] = zCurrent;
            zCurrent += obj[segName]['length'];
            obj[segName]['endPos'] = zCurrent;
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
                const segName = Object.keys(segment)[0];
                segment[segName][priv] = beamSegmentInfo[segName][priv];
            }
        }
        return segment
    };

    //  Handles both color and start and end pos for ENTIRE beamline
    const setSelectedItemsHandler = (segList) => {
        const cleanedSegList = segList.map((segment) => {
            return handleSegmentColor(segment);
        });

        calcStartEndPos(cleanedSegList);
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

        console.log(grouped);
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

    const handleItemClick = (item) => {
        const beamObj = handleSegmentColor({[item]: structuredClone(beamSegmentInfo[item])});
        //console.log('updated Obj', beamObj);
        const updatedList = [...beamlistSelected, beamObj];
        calcStartEndPos(updatedList);
    };

    const handleDelete = (index) => {
        const updatedBeamline = beamlistSelected.filter((_, i) => i !== index);
        calcStartEndPos(updatedBeamline);
    };

    const handleParamChange = (index, paramKey, newValue) => {
        const updatedList = (
            beamlistSelected.map((item, i) => {
                if (i !== index) return item;
                const topKey = Object.keys(item)[0];
                const updatedParams = {
                    ...item[topKey],
                    [paramKey]: parseFloat(newValue)
                };
    
                return {
                    [topKey]: updatedParams
                };
            })
        );
        calcStartEndPos(updatedList);
    };


    const getBeamline = async (segList) => {
        const uiErrorStatus = errorCatcher();
        if (uiErrorStatus) {
            return
        };
        const cleanedList = segList.map(obj => {
            const key = Object.keys(obj)[0];
            const cleanedParams = Object.fromEntries(
              Object.entries(obj[key]).filter(([p]) => !PRIVATEVARS.includes(p))
            );
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
    
        const jsonBody = JSON.stringify(plottingParams, null, 4); 
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



    return (
        <>
        <ErrorWindow message={errorMessage}
                     showError = {showError} /> 
        <div className="layout">
          <div className="sidebar">
            <h2>FEL simulator</h2>
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
            <h3>Beam setup</h3>
            <div className="scrollBox">
                {beamlistSelected.map((item, index) => (
                    <BeamSegment 
                            key={index}
                            name={Object.keys(item)[0]}
                            params={item}
                            index={index}
                            onDelete={handleDelete}
                            onChanges={handleParamChange}
                            PRIVATEVARS={PRIVATEVARS}
                    />
                ))}
                {/*<button type="button"
                        className="editTableButton">
                        Edit Table
                </button>*/}
            </div>
          </div>
          <div className="beamSettings">
            <ExcelUploadButton excelToAPI={excelToAPI} />
            <label htmlFor="beamtypeSelect" className="forLabels">Select Beam type:</label>
            <select name="beamtypeSelect" onChange={(e) => setBeamInput(e.target.value)}>
                <option value="electron">Electron</option>
                <option value="proton">Proton</option>
                <option value="otherIon">Other Ion</option>
            </select>
            {
                (currentBeamType !== "electron" && currentBeamType !== "proton") && (
                <input
                    type="text"
                    onChange={(e) => setBeamtypeToPass(e.target.value)}
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
          <div className="main-content">
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
