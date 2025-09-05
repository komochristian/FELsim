import React, {useState, useEffect} from 'react';
import * as d3 from 'd3';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import BeamSegment from './components/BeamSegment/BeamSegment';
import BeamlineScroll from './components/BeamlineScroll/BeamlineScroll';
import DiscreteSlider from './components/DiscreteSlider/DiscreteSlider';
import ExcelUploadButton from './components/ExcelUploadButton/ExcelUploadButton';
import LineGraph from './components/LineGraph/LineGraph';

function App()
{
    const PRIVATEVARS = ['color', 'startPos', 'endPos'];  // USE THIS SO USERS CANT EDIT THESE VALUES
    const API_ROUTE = import.meta.env.VITE_DOCKER_ROUTE || 'http://0.0.0.0:8000';
    //console.log(API_ROUTE);

    const [beamSegmentInfo, setData] = useState(null);
    const [dotGraphs, setDotGraphs] = useState([]);
    const [lineGraph, setLineGraph] = useState(null);
    const [beamlistSelected, setSelectedItems] = useState([]);
    const [currentZ, setZValue] = useState(0);
    const [currentBeamType, setBeamType] = useState('electron');
    const [twissDf, setTwissDf] = useState([]);
    const [totalLen, setTotalLen] = useState(0);
    const [numOfParticles, setParticleNum] = useState(1000);

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
        const twissPlotData = Object.entries(twissJsonObj).flatMap(([key, obj]) => {
                return Object.entries(obj).map(([axis, arr]) => {
                
                    return {"id": `${key}: ${axis}`,
                            "data": 
                                arr.map((val, i) => ({
                                    'x': x_axis[i],
                                    'y': val
                                    })
                                )
                    }
                });
        });
        
        setTwissDf(twissPlotData);
    };

    const excelToAPI = async (fileJSON) => {  
        const res =  await fetch(API_ROUTE + '/excel-to-beamline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(fileJSON, null, 2),
        });
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
            beamType: currentBeamType
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
                </div>
          </div>
          <div className="beamSettings">
            <ExcelUploadButton excelToAPI={excelToAPI} />
            <label htmlFor="beamtypeSelect" className="forLabels">Select Beam type:</label>
            <select name="beamtypeSelect" onChange={(e) => setBeamType(e.target.value)}>
                <option value="electron">Electron</option>
                <option value="proton">Proton</option>
                <option value="otherIon">Other Ion</option>
            </select>
            {
                (!['electron', 'proton'].includes(currentBeamType)) && (
                <input
                    type="text"
                    onChange={(e) => setBeamType(e.target.value)}
                />)
            }
            <label htmlFor="numParticles">Number of particles:</label>
            <input defaultValue={numOfParticles}
                   type="number"
                    name="numParticles" 
                    onChange={(e) => setParticleNum(e.target.value)}
                    min={3}
            />
            
          </div>
          <div className="main-content">
                <img src={dotGraphs.size > 0 ? dotGraphs.get(currentZ) : null} alt="loading..."/>
          </div>
          <div className="twiss-graph">
                <LineGraph twissData={twissDf}
                           setZValue={setZValue} 
                           beamline={beamlistSelected}
                           totalLen={totalLen}>
                </LineGraph>
            {/*<img src={lineGraph ? lineGraph: null} alt="loading..."/>*/}
          </div>
          
        </div>
        </>
    );
}

export default App;
