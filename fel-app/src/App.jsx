import React, {useState, useEffect} from 'react';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import BeamSegment from './components/BeamSegment/BeamSegment';
function App()
{
    const [beamSegmentInfo, setData] = useState(null);
    const [dotGraphs, setDotGraphs] = useState([]);
    const [lineGraph, setLineGraph] = useState(null);
    const [selectedItems, setSelectedItems] = useState([]);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/get-beamsegmentinfo')
            .then((response) => response.json())
            .then((json) => setData(json))
            .catch((err) => console.error("Error loading beam segment info:", err));
        }, []);


    useEffect(() => {
        console.log("Updated selectedItems:", selectedItems);
    }, [selectedItems]);

    if (!beamSegmentInfo) return <div>Loading...</div>;
    const items = Object.keys(beamSegmentInfo);

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
        console.log("json sent;", jsonBody);

        const res = await fetch('http://127.0.0.1:8000/load-axes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: jsonBody,
        });
        
        const axImages = await res.json();
        const result = axImages['images'];
        const lineAx = axImages['line-graph'];
        const cleanResult = result.map(axis => {
            return `data:image/png;base64,${axis}`
        });
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
                <img src={dotGraphs.length > 0 ? dotGraphs[0] : null} alt="loading..."/>
          </div>
          <div className="linegraph ">
            <img src={lineGraph ? lineGraph: null} alt="loading..."/>
          </div>
          <div className="twiss">
            <h6>Twiss options</h6>
          </div>
        </div>
        </>
    );
}

export default App;
