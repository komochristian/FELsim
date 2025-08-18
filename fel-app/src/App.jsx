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
    const [beamSegmentInfo, setData] = useState(null);
    const [dotGraphs, setDotGraphs] = useState([]);
    const [lineGraph, setLineGraph] = useState(null);
    const [selectedItems, setSelectedItems] = useState([]);
    const [currentZ, setZValue] = useState(0);
    const [currentBeamType, setBeamType] = useState(null);
    const [twissDf, setTwissDf] = useState([]);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/beamsegmentinfo')
            .then((response) => response.json())
            .then((json) => setData(json))
            .catch((err) => console.error("Error loading beam segment info:", err));
        }, []);
    console.log(beamSegmentInfo);

    useEffect(() => {
        console.log("Updated selectedItems:", selectedItems);
    }, [selectedItems]);

    if (!beamSegmentInfo) return <div>Loading...</div>;
    const items = Object.keys(beamSegmentInfo);

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
        const res = await fetch('http://127.0.0.1:8000/excel-to-beamline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(fileJSON, null, 2),
        });
        const beamlist = await res.json();
        console.log(beamlist);
        setSelectedItems(beamlist);
    };

    const handleItemClick = (item) => {
        setSelectedItems(prevItems => [...prevItems, {[item]: beamSegmentInfo[item]}]);
    };

    const handleDelete = (index) => {
        setSelectedItems(prev => prev.filter((_, i) => i !== index));
    };

    const handleParamChange = (index, paramKey, newValue) => {
        setSelectedItems(prev =>
            prev.map((item, i) => {
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
        )
    };

    const getBeamline = async (segList) => {
        const cleanedList = segList.map(obj => {
            const key = Object.keys(obj)[0];
            return {
                segmentName: key,
                parameters: obj[key]
            };
        });
    
        const jsonBody = JSON.stringify(cleanedList, null, 4); 
        //console.log("json sent;", jsonBody);

        const res = await fetch('http://127.0.0.1:8000/axes', {
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
                onClick={() => getBeamline(selectedItems)}>
                Simulate
            </button>
            <ExcelUploadButton excelToAPI={excelToAPI} />
            <label htmlFor="beamtypeSelect" className="forLabels">Select Beam type:</label>
            <select name="beamtypeSelect">
                <option value="electron">Electron</option>
                <option value="proton">Proton</option>
                <option value="otherIon">Other Ion</option>
            </select>
            <h3>Beam setup</h3>
               <div className="scrollBox">
                    {selectedItems.map((item, index) => (
                        <BeamSegment 
                            key={index}
                            name={Object.keys(item)[0]}
                            params={item}
                            index={index}
                            onDelete={handleDelete}
                            onChanges={handleParamChange}
                        />
                    ))}
                </div>
          </div>
          <div className="main-content">
                <img src={dotGraphs.size > 0 ? dotGraphs.get(currentZ) : null} alt="loading..."/>
          </div>
          <div className="twiss-graph">
                <LineGraph twissData={twissDf} setZValue={setZValue}></LineGraph>
            {/*<img src={lineGraph ? lineGraph: null} alt="loading..."/>*/}
          </div>
          
        </div>
        </>
    );
}

export default App;
